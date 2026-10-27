#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
import cv2
from pathlib import Path
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
from PIL import Image

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def edge_aware_postprocess_mesh(mesh, viewpoint_stack, mask_dir=None, mask_pattern="mask_{stem}.png", shrink_ratio=0.15):
    """Conservative mesh shrinkage guided by projected foreground silhouette."""
    if shrink_ratio <= 0.0 or mask_dir is None:
        return mesh, None

    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return mesh, None

    keep_votes = np.zeros((vertices.shape[0],), dtype=np.float32)
    valid_votes = np.zeros_like(keep_votes)
    used_masks = 0

    for cam in viewpoint_stack:
        stem = os.path.splitext(cam.image_name)[0]
        mask_path = os.path.join(mask_dir, mask_pattern.format(stem=stem, image_name=cam.image_name))
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        used_masks += 1
        mask = cv2.resize(mask, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)

        verts_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
        full_proj = cam.full_proj_transform.detach().cpu().numpy()
        clip = verts_h @ full_proj
        z = clip[:, 3]
        valid = z > 1e-6
        if not np.any(valid):
            continue
        ndc = clip[:, :2] / z[:, None]
        px = ((ndc[:, 0] + 1.0) * 0.5 * (cam.image_width - 1)).round().astype(np.int32)
        py = ((ndc[:, 1] + 1.0) * 0.5 * (cam.image_height - 1)).round().astype(np.int32)
        in_view = (
            valid
            & (px >= 0)
            & (px < cam.image_width)
            & (py >= 0)
            & (py < cam.image_height)
        )
        valid_votes[in_view] += 1.0
        keep_votes[in_view] += (mask[py[in_view], px[in_view]] > 127).astype(np.float32)

    vote_mask = valid_votes > 0
    if not np.any(vote_mask):
        return mesh, {
            "used_masks": used_masks,
            "projected_vertices": 0,
            "outside_ratio_before": None,
            "mean_consistency": None,
            "mean_shrink_scale": 1.0,
            "mean_displacement": 0.0,
            "max_displacement": 0.0,
        }

    consistency = np.ones_like(keep_votes)
    consistency[vote_mask] = keep_votes[vote_mask] / np.maximum(valid_votes[vote_mask], 1.0)

    center = vertices.mean(axis=0, keepdims=True)
    direction = vertices - center
    scale = (1.0 - shrink_ratio * np.clip(1.0 - consistency, 0.0, 1.0))[:, None]
    new_vertices = center + direction * scale
    displacement = np.linalg.norm(new_vertices - vertices, axis=1)
    metrics = {
        "used_masks": used_masks,
        "projected_vertices": int(vote_mask.sum()),
        "outside_ratio_before": float(np.mean(1.0 - consistency[vote_mask])),
        "mean_consistency": float(np.mean(consistency[vote_mask])),
        "mean_shrink_scale": float(np.mean(scale[vote_mask])),
        "mean_displacement": float(np.mean(displacement[vote_mask])),
        "max_displacement": float(np.max(displacement[vote_mask])),
    }

    mesh_out = o3d.geometry.TriangleMesh(mesh)
    mesh_out.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh_out.compute_vertex_normals()
    return mesh_out, metrics

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # §2.8 needs the rendered surface normal for w_normal, so it is kept.
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            # self.alphamaps.append(alpha.cpu())
            # self.normals.append(normal.cpu())
            # §2.8 w_normal needs the rendered surface normal per view.
            self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def _fusion_grid(self, voxel_size, depth_trunc, padding=0.0):
        """Voxel-centre grid covering the point cloud, plus world->voxel helpers.

        The grid is anchored on the Gaussian centres so it is independent of the
        camera bounding sphere: a foreground-object cloud has a much smaller
        extent than the training cameras, and anchoring on the cameras wastes
        most of the volume on empty background.
        """
        pts = self.gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        lo = pts.min(axis=0) - padding
        hi = pts.max(axis=0) + padding
        span = np.maximum(hi - lo, voxel_size)
        dims = np.maximum(np.ceil(span / voxel_size).astype(int) + 1, 2)
        return lo, hi, dims

    @torch.no_grad()
    def extract_mesh_weighted(
        self,
        voxel_size=0.004,
        sdf_trunc=0.02,
        depth_trunc=3,
        mask_background=True,
        mask_dir="",
        mask_pattern="mask_{stem}.png",
        normal_sharpness=8.0,
        vis_weight_min=0.2,
        normal_weight_min=0.2,
        mask_weight_min=0.0,
        chunk_size=200000,
        export_diagnostics=None,
    ):
        """Manuscript §2.8: visibility-, normal- and mask-aware weighted TSDF.

            D(x) = sum_c w_c(x) d_c(x) / (sum_c w_c(x) + eps)
            w_c(x) = w_c^vis(x) * w_c^normal(x) * w_c^mask(x)

        Every factor is a *support* weight in [0, 1] and none of them is a hard
        cut for a view that genuinely observes the voxel:

        w^vis     confidence that this view sees x near its rendered surface.
                  w^vis = clamp(1 - |d_c(x)| / sdf_trunc, vis_weight_min, 1) for
                  voxels inside the truncation band; views whose depth range does
                  not reach x contribute nothing.  Uniform integration (Open3D)
                  instead lets one grazing view outweigh twenty head-on ones.
        w^normal  alignment of the *rendered* surface normal at the projected
                  pixel with the view ray:
                  w^normal = normal_weight_min + (1 - normal_weight_min) *
                             |n . v|^normal_sharpness .
                  Oblique observations have the least reliable depth, so they are
                  down-weighted rather than discarded.
        w^mask    the RAP-FSAM3 foreground prior sampled bilinearly at the
                  projected pixel: 1 inside the potted-plant mask,
                  `mask_weight_min` outside (default 0 = the view cannot support
                  a voxel that falls outside its own foreground prior).

        The grid is anchored on the Gaussian centres, so the volume covers the
        reconstructed object rather than the (much larger) camera rig.  Colour is
        fused with the same weights, so the exported mesh keeps a texture.

        `export_diagnostics` (a dict) receives the fusion statistics.
        """
        eps = 1e-8
        lo, hi, dims = self._fusion_grid(voxel_size, depth_trunc)
        nx, ny, nz = (int(d) for d in dims)
        n_vox = nx * ny * nz
        print(f"[weighted TSDF] grid {nx}x{ny}x{nz} = {n_vox} voxels @ {voxel_size}")

        xx = (lo[0] + (np.arange(nx, dtype=np.float32) + 0.5) * voxel_size)
        yy = (lo[1] + (np.arange(ny, dtype=np.float32) + 0.5) * voxel_size)
        zz = (lo[2] + (np.arange(nz, dtype=np.float32) + 0.5) * voxel_size)
        gx, gy, gz = np.meshgrid(xx, yy, zz, indexing="ij")
        grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        del gx, gy, gz

        tsdf = np.full((n_vox,), -1.0, dtype=np.float32)
        wsum = np.zeros((n_vox,), dtype=np.float32)
        rgb_acc = np.zeros((n_vox, 3), dtype=np.float32)
        view_count = np.zeros((n_vox,), dtype=np.float32)
        weight_sum_per_cue = {"vis": 0.0, "normal": 0.0, "mask": 0.0,
                              "combined": 0.0}
        n_contrib = 0

        cams = to_cam_open3d(self.viewpoint_stack)
        diag = {"views": len(self.viewpoint_stack), "grid": [nx, ny, nz],
                "voxel_size": float(voxel_size), "sdf_trunc": float(sdf_trunc),
                "depth_trunc": float(depth_trunc), "per_view": []}

        def _load_mask(cam):
            if not mask_dir:
                return None
            stem = Path(getattr(cam, "image_name", "") or "").stem
            if not stem:
                return None
            path = Path(mask_dir) / mask_pattern.format(stem=stem)
            if not path.exists():
                return None
            return np.array(Image.open(path).convert("L")) > 127

        for i, cam_o3d in enumerate(tqdm(cams, desc="weighted TSDF integration")):
            cam = self.viewpoint_stack[i]
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            normal = self.depth_normals[i] if getattr(self, "depth_normals", None) else None
            h, w = int(depth.shape[-2]), int(depth.shape[-1])
            intr = cam_o3d.intrinsic.intrinsic
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            world = cam_o3d.extrinsic                      # x_cam = R x_world + t
            R = torch.from_numpy(world[:3, :3].astype(np.float32))
            t = torch.from_numpy(world[:3, 3].astype(np.float32))

            fg_mask = _load_mask(cam)
            if fg_mask is not None and fg_mask.shape[:2] != (h, w):
                fg_mask = None              # mask resolution must match the render
            mask_t = (torch.from_numpy(fg_mask.astype(np.float32))[None, None]
                      if fg_mask is not None else None)
            depth_t = depth.reshape(1, 1, h, w).float()
            rgb_t = rgb.reshape(1, 3, h, w).float()
            if normal is not None:
                # `surf_normal` is rendered in view space; rotate to world space.
                n_flat = normal.reshape(3, -1).permute(1, 0).float()
                n_world = n_flat @ R.T
                n_world = n_world / n_world.norm(dim=1, keepdim=True).clamp_min(eps)
            else:
                n_world = None

            view_samples = 0
            for start in range(0, n_vox, chunk_size):
                pts = torch.from_numpy(grid[start:start + chunk_size])
                cam_pts = pts @ R.T + t                    # (M, 3) in camera space
                z = cam_pts[:, 2]
                in_front = (z > 1e-6) & (z < depth_trunc)
                if not bool(in_front.any()):
                    continue
                pix_u = fx * cam_pts[:, 0] / z.clamp_min(eps) + cx
                pix_v = fy * cam_pts[:, 1] / z.clamp_min(eps) + cy
                inside = (in_front
                          & (pix_u >= 0) & (pix_u <= w - 1)
                          & (pix_v >= 0) & (pix_v <= h - 1))
                if not bool(inside.any()):
                    continue
                sel = torch.nonzero(inside, as_tuple=False).reshape(-1)
                gx_n = 2.0 * pix_u[sel] / max(w - 1, 1) - 1.0
                gy_n = 2.0 * pix_v[sel] / max(h - 1, 1) - 1.0
                sample_grid = torch.stack([gx_n, gy_n], dim=1).reshape(1, 1, -1, 2)

                d_sampled = torch.nn.functional.grid_sample(
                    depth_t, sample_grid, mode="bilinear", padding_mode="zeros",
                    align_corners=True).reshape(-1)
                sdf = d_sampled - z[sel]
                band = (d_sampled > 0) & (sdf.abs() <= sdf_trunc)
                if not bool(band.any()):
                    continue
                sel2 = sel[band]
                sdf = sdf[band]

                w_vis = (1.0 - sdf.abs() / max(sdf_trunc, eps)).clamp(
                    float(vis_weight_min), 1.0)
                if n_world is not None:
                    n_pix = torch.nn.functional.grid_sample(
                        n_world.T.reshape(1, 3, h, w),
                        sample_grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True).reshape(3, -1).T[band]
                    n_pix = n_pix / n_pix.norm(dim=1, keepdim=True).clamp_min(eps)
                    v = cam_pts[sel2]
                    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)
                    align = (n_pix * v).sum(dim=1).abs().clamp(0.0, 1.0)
                    w_normal = float(normal_weight_min) + (
                        1.0 - float(normal_weight_min)) * align.pow(float(normal_sharpness))
                else:
                    w_normal = torch.ones_like(w_vis)
                if mask_t is not None:
                    m_pix = torch.nn.functional.grid_sample(
                        mask_t, sample_grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True).reshape(-1)[band]
                    w_mask = m_pix * (1.0 - float(mask_weight_min)) + float(mask_weight_min)
                else:
                    w_mask = torch.ones_like(w_vis)

                w_total = (w_vis * w_normal * w_mask).clamp_min(0.0)
                keep = w_total > eps
                if not bool(keep.any()):
                    continue
                sel3 = sel2[keep]
                sdf3 = sdf[keep]
                w3 = w_total[keep]
                cue = w_vis[keep]
                cue_n = w_normal[keep]
                cue_m = w_mask[keep]

                idx_np = (start + sel3).cpu().numpy()
                sdf_np = (sdf3 / max(sdf_trunc, eps)).clamp(-1.0, 1.0).cpu().numpy()
                w_np = w3.cpu().numpy()
                prev = wsum[idx_np]
                new_w = prev + w_np
                tsdf[idx_np] = (tsdf[idx_np] * prev + sdf_np * w_np) / np.maximum(new_w, eps)
                wsum[idx_np] = new_w
                view_count[idx_np] += 1.0
                pix_col = torch.nn.functional.grid_sample(
                    rgb_t, sample_grid, mode="bilinear", padding_mode="zeros",
                    align_corners=True).reshape(3, -1).T[band][keep]
                rgb_acc[idx_np] = rgb_acc[idx_np] + pix_col.cpu().numpy() * w_np[:, None]
                weight_sum_per_cue["vis"] += float(cue.sum())
                weight_sum_per_cue["normal"] += float(cue_n.sum())
                weight_sum_per_cue["mask"] += float(cue_m.sum())
                weight_sum_per_cue["combined"] += float(w3.sum())
                n_contrib += int(w3.numel())
                view_samples += int(w3.numel())

            diag["per_view"].append({"view": i, "view_weight_samples": view_samples})

        occupied = wsum > eps
        n_occ = int(occupied.sum())
        if n_occ:
            rgb_acc[occupied] /= wsum[occupied][:, None]
        diag["occupied_voxels"] = n_occ
        diag["mean_effective_views"] = (
            float(view_count[occupied].mean()) if n_occ else 0.0)
        diag["mean_fused_weight"] = (
            float(wsum[occupied].mean()) if n_occ else 0.0)
        diag["cue_mean"] = {
            k: (v / max(n_contrib, 1)) for k, v in weight_sum_per_cue.items()}
        print(f"[weighted TSDF] occupied {n_occ} voxels, "
              f"mean effective views {diag['mean_effective_views']:.2f}, "
              f"cue means {diag['cue_mean']}")

        if isinstance(export_diagnostics, dict):
            export_diagnostics.update(diag)

        return self._marching_cubes_weighted(
            tsdf.reshape(nx, ny, nz), wsum.reshape(nx, ny, nz),
            rgb_acc.reshape(nx, ny, nz, 3), lo, voxel_size)

    @torch.no_grad()
    def _marching_cubes_weighted(self, tsdf, weights, colors, lo, voxel_size):
        """Marching cubes on the weighted TSDF, keeping the fused colour.

        Voxels outside every view's truncation band carry no distance estimate.
        They are *sign-extrapolated* from the nearest supported voxel instead of
        being forced to a constant: forcing them to +1 would place a spurious
        zero crossing on the band's own boundary, which shows up as a shell one
        truncation width off the true surface.
        """
        from skimage import measure
        supported = weights > 1e-8
        if not supported.any():
            print("[weighted TSDF] no voxel received support; empty mesh")
            return o3d.geometry.TriangleMesh()
        field = np.ones_like(tsdf, dtype=np.float32)
        field[supported] = tsdf[supported]
        # Extrapolate along the depth axis of each voxel column: an unsupported
        # voxel in a column that has support takes the sign of the nearest
        # supported voxel *in its own column*.  A 3D nearest-neighbour fill is not
        # safe here -- it can pick a voxel on the far side of the surface and
        # create a spurious crossing one truncation width off the real surface.
        nx, ny, nz = tsdf.shape
        cols = field.reshape(-1, nz)
        col_sup = supported.reshape(-1, nz)
        has = col_sup.any(axis=1)
        if has.any():
            k = np.arange(nz)[None, :]
            below = np.where(col_sup, k, -1)
            below = np.maximum.accumulate(below, axis=1)
            above = np.where(col_sup, k, nz)
            above = np.minimum.accumulate(above[:, ::-1], axis=1)[:, ::-1]
            safe_below = np.clip(below, 0, nz - 1)
            safe_above = np.clip(above, 0, nz - 1)
            val_below = np.take_along_axis(cols, safe_below, axis=1)
            val_above = np.take_along_axis(cols, safe_above, axis=1)
            pick_below = (below >= 0) & ((above >= nz) | ((k - safe_below) <= (safe_above - k)))
            fill = np.where(pick_below, val_below, val_above)
            filled = np.where(col_sup, cols, np.where(fill >= 0.0, 1.0, -1.0))
            cols[has] = filled[has]
        field = cols.reshape(nx, ny, nz)
        verts, faces, _, _ = measure.marching_cubes(
            field, level=0.0, spacing=(float(voxel_size),) * 3)
        verts = verts + np.asarray(lo, dtype=np.float32)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)))
        if colors is not None and np.any(colors):
            # Nearest occupied voxel colour for each vertex (cheap and stable).
            coords = np.clip(((verts - np.asarray(lo)) / voxel_size).round().astype(int),
                             0, np.array(colors.shape[:3]) - 1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(colors[coords[:, 0], coords[:, 1], coords[:, 2]], 0.0, 1.0))
        mesh.compute_vertex_normals()
        return mesh

    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
