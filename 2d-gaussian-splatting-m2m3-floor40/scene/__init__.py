#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.plant_aware_utils import dilate_mask, enabled, filter_by_name_set, load_mask_tensor, read_name_set


def _foreground_filter_point_cloud(point_cloud, scene_info, cam_infos, args):
    mode = getattr(args, "init_pcd_mode", "none")
    if point_cloud is None or mode not in ("foreground_mask", "foreground_track"):
        return point_cloud, None
    if not enabled(getattr(args, "mask_dir", "")):
        raise ValueError("--init_pcd_mode foreground_mask/foreground_track requires --mask_dir <path>.")

    if mode == "foreground_track":
        return _foreground_filter_point_cloud_by_track(point_cloud, scene_info, args)

    usable_cams = list(cam_infos)
    max_cameras = int(getattr(args, "init_pcd_max_cameras", 0))
    if max_cameras > 0:
        stride = max(1, len(usable_cams) // max_cameras)
        usable_cams = usable_cams[::stride][:max_cameras]
    if not usable_cams:
        raise ValueError("Foreground point-cloud filtering has no usable cameras.")

    xyz = torch.from_numpy(np.asarray(point_cloud.points)).float().cuda()
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    obs_count = torch.zeros((xyz.shape[0],), dtype=torch.int16, device="cuda")
    fg_count = torch.zeros_like(obs_count)
    chunk_size = int(getattr(args, "init_pcd_chunk_size", 200000))
    dilate_px = int(getattr(args, "init_pcd_dilate_mask_px", 0))

    for cam_idx, cam in enumerate(usable_cams, start=1):
        world_view = torch.tensor(cam.world_view_transform, dtype=torch.float32, device="cuda")
        projection = torch.tensor(cam.projection_matrix, dtype=torch.float32, device="cuda")
        full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        mask = load_mask_tensor(
            args.mask_dir,
            cam.image_name,
            (cam.image_width, cam.image_height),
            pattern=getattr(args, "mask_pattern", "mask_{stem}.png"),
            threshold=getattr(args, "mask_threshold", 0.5),
        ).cuda()
        if dilate_px > 0:
            mask = dilate_mask(mask, dilate_px)

        for start in range(0, xyz.shape[0], chunk_size):
            end = min(start + chunk_size, xyz.shape[0])
            projected = xyz_h[start:end] @ full_proj
            z = projected[:, 3]
            pix = projected[:, :2] / z[:, None].clamp_min(1e-6)
            in_view = (
                (z > 0)
                & (pix[:, 0] > -1.0)
                & (pix[:, 0] < 1.0)
                & (pix[:, 1] > -1.0)
                & (pix[:, 1] < 1.0)
            )
            if not torch.any(in_view):
                continue
            sampled = torch.nn.functional.grid_sample(
                mask[None],
                pix[None, None],
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(-1)
            obs_count[start:end] += in_view.to(torch.int16)
            fg_count[start:end] += ((sampled > 0.5) & in_view).to(torch.int16)
        if cam_idx % 25 == 0 or cam_idx == len(usable_cams):
            print(f"[Plant-aware] foreground init PCD voted {cam_idx}/{len(usable_cams)} cameras")

    min_obs = int(getattr(args, "init_pcd_min_observations", 3))
    threshold = float(getattr(args, "init_pcd_foreground_threshold", 0.5))
    visible = obs_count >= min_obs
    fg_ratio = torch.zeros((xyz.shape[0],), dtype=torch.float32, device="cuda")
    fg_ratio[visible] = fg_count[visible].float() / obs_count[visible].float().clamp_min(1.0)
    keep = visible & (fg_ratio >= threshold)
    if not torch.any(keep):
        raise ValueError("Foreground point-cloud filtering removed all points; relax thresholds.")

    keep_np = keep.detach().cpu().numpy()
    report = {
        "mode": "foreground_mask",
        "points_before": int(xyz.shape[0]),
        "points_after": int(keep_np.sum()),
        "kept_ratio": float(keep_np.mean()),
        "num_cameras_used": len(usable_cams),
        "min_observations": min_obs,
        "foreground_threshold": threshold,
        "dilate_mask_px": dilate_px,
    }
    filtered = BasicPointCloud(
        points=np.asarray(point_cloud.points)[keep_np],
        colors=np.asarray(point_cloud.colors)[keep_np],
        normals=np.asarray(point_cloud.normals)[keep_np],
    )
    return filtered, report


def _foreground_filter_point_cloud_by_track(point_cloud, scene_info, args):
    if scene_info.point3d_ids is None or scene_info.point3d_tracks is None:
        raise ValueError("foreground_track requires COLMAP points3D.bin tracks.")

    from PIL import Image

    mask_cache = {}
    obs_count = np.zeros((len(scene_info.point3d_ids),), dtype=np.int32)
    fg_count = np.zeros_like(obs_count)
    dilate_px = int(getattr(args, "init_pcd_dilate_mask_px", 0))

    def get_mask(image_name):
        stem = os.path.splitext(os.path.basename(image_name))[0]
        if stem in mask_cache:
            return mask_cache[stem]
        mask = load_mask_tensor(
            args.mask_dir,
            stem,
            Image.open(os.path.join(args.source_path, "images", image_name)).size,
            pattern=getattr(args, "mask_pattern", "mask_{stem}.png"),
            threshold=getattr(args, "mask_threshold", 0.5),
        )
        if dilate_px > 0:
            mask = dilate_mask(mask, dilate_px)
        mask_cache[stem] = mask[0].cpu().numpy() > 0.5
        return mask_cache[stem]

    for idx, point_id in enumerate(scene_info.point3d_ids):
        for image_name, x, y in scene_info.point3d_tracks.get(int(point_id), []):
            mask = get_mask(image_name)
            u = int(round(x))
            v = int(round(y))
            if u < 0 or v < 0 or v >= mask.shape[0] or u >= mask.shape[1]:
                continue
            obs_count[idx] += 1
            if mask[v, u]:
                fg_count[idx] += 1

    min_obs = int(getattr(args, "init_pcd_min_observations", 3))
    threshold = float(getattr(args, "init_pcd_foreground_threshold", 0.5))
    visible = obs_count >= min_obs
    fg_ratio = np.zeros_like(obs_count, dtype=np.float32)
    fg_ratio[visible] = fg_count[visible] / np.maximum(obs_count[visible], 1)
    keep_np = visible & (fg_ratio >= threshold)
    if not np.any(keep_np):
        raise ValueError("Foreground track point-cloud filtering removed all points; relax thresholds.")

    report = {
        "mode": "foreground_track",
        "points_before": int(len(scene_info.point3d_ids)),
        "points_after": int(keep_np.sum()),
        "kept_ratio": float(keep_np.mean()),
        "min_observations": min_obs,
        "foreground_threshold": threshold,
        "dilate_mask_px": dilate_px,
        "num_masks_loaded": len(mask_cache),
    }
    filtered = BasicPointCloud(
        points=np.asarray(point_cloud.points)[keep_np],
        colors=np.asarray(point_cloud.colors)[keep_np],
        normals=np.asarray(point_cloud.normals)[keep_np],
    )
    return filtered, report

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        for gate_name, list_attr in (
            ("raw", "raw_gate_list"),
            ("mask", "mask_gate_list"),
            ("geo", "geo_gate_list"),
        ):
            list_path = getattr(args, list_attr, "")
            if enabled(list_path):
                keep_names = read_name_set(list_path)
                before_train = len(scene_info.train_cameras)
                before_test = len(scene_info.test_cameras)
                scene_info = scene_info._replace(
                    train_cameras=filter_by_name_set(scene_info.train_cameras, keep_names),
                    test_cameras=filter_by_name_set(scene_info.test_cameras, keep_names),
                )
                print(
                    f"[Plant-aware] {gate_name} gate retained "
                    f"{len(scene_info.train_cameras)}/{before_train} train and "
                    f"{len(scene_info.test_cameras)}/{before_test} test cameras"
                )
                if not scene_info.train_cameras:
                    raise ValueError(f"{gate_name} gate retained no training cameras: {list_path}")

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            init_pcd, init_report = _foreground_filter_point_cloud(
                scene_info.point_cloud,
                scene_info,
                self.train_cameras[resolution_scales[0]],
                args,
            )
            if init_report is not None:
                with open(os.path.join(self.model_path, "foreground_init_pcd_report.json"), "w", encoding="utf-8") as f:
                    json.dump(init_report, f, indent=2, ensure_ascii=False)
                print(
                    "[Plant-aware] foreground init PCD kept "
                    f"{init_report['points_after']}/{init_report['points_before']} "
                    f"points ({init_report['kept_ratio']:.2%})"
                )
            self.gaussians.create_from_pcd(init_pcd, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
