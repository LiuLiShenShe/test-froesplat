# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import zipfile
import copy
import time

import cv2
import numpy as np
import pycolmap
import requests
import trimesh
from matplotlib import colormaps
from PIL import Image
from scipy.spatial.transform import Rotation


def _patch_trimesh_gltf_point_normals():
    """Ensure trimesh GLTF exporter includes NORMAL attribute for PointCloud objects if present."""
    import trimesh.exchange.gltf
    if getattr(trimesh.exchange.gltf, "_normals_patched", False):
        return

    original_append_point = trimesh.exchange.gltf._append_point

    def custom_append_point(points, name, tree, buffer_items):
        original_append_point(points, name, tree, buffer_items)
        normals = getattr(points, "normals", None)
        if normals is None and hasattr(points, "_data") and "normals" in points._data:
            normals = points._data["normals"]
        if normals is not None and len(normals) == len(points.vertices):
            acc_normal = trimesh.exchange.gltf._data_append(
                acc=tree["accessors"],
                buff=buffer_items,
                blob={"componentType": 5126, "type": "VEC3", "byteOffset": 0},
                data=np.ascontiguousarray(normals, dtype=np.float32),
            )
            tree["meshes"][-1]["primitives"][0]["attributes"]["NORMAL"] = acc_normal

    trimesh.exchange.gltf._append_point = custom_append_point
    trimesh.exchange.gltf._normals_patched = True


_patch_trimesh_gltf_point_normals()


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: str | None = None,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
    split_layers: bool = False,
) -> trimesh.Scene:
    """Convert VGGT-Omega camera/depth predictions to a GLB scene."""
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    per_frame_points = _get_filtered_points(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        max_points=max_points,
        filter_depth_edges=filter_depth_edges,
        depth_edge_rtol=depth_edge_rtol,
    )

    all_verts_list = [v for v, c, n in per_frame_points if v.size > 0]
    if len(all_verts_list) == 0:
        scene_scale = 1.0
    else:
        all_vertices = np.concatenate(all_verts_list, axis=0)
        lower = np.percentile(all_vertices, 5, axis=0)
        upper = np.percentile(all_vertices, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper - lower))
        if scene_scale <= 0:
            scene_scale = 1.0

    scene = trimesh.Scene()
    num_frames = len(per_frame_points)

    if split_layers:
        # One PointCloud per camera, named individually
        for i, (vertices, colors, normals) in enumerate(per_frame_points):
            if vertices.size > 0:
                pc = trimesh.PointCloud(vertices=vertices, colors=colors)
                if normals is not None and len(normals) == len(vertices):
                    pc._data["normals"] = normals
                layer_name = f"cam_{i+1:02d}_points" if num_frames >= 10 else f"cam_{i+1}_points"
                scene.add_geometry(pc, node_name=layer_name, geom_name=layer_name)
    else:
        # Merge all points into a single PointCloud layer
        all_verts = [v for v, c, n in per_frame_points if v.size > 0]
        all_cols = [c for v, c, n in per_frame_points if v.size > 0]
        all_norms = [n for v, c, n in per_frame_points if v is not None and v.size > 0]
        if all_verts:
            merged_verts = np.concatenate(all_verts, axis=0)
            merged_cols = np.concatenate(all_cols, axis=0)
            pc = trimesh.PointCloud(vertices=merged_verts, colors=merged_cols)
            if all(n is not None and len(n) > 0 for n in all_norms):
                pc._data["normals"] = np.concatenate(all_norms, axis=0)
            scene.add_geometry(pc, node_name="points", geom_name="points")

    camera_matrices = predictions["extrinsic"]
    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices
    extrinsics[:, 3, 3] = 1.0

    if show_cam:
        colormap = colormaps.get_cmap("gist_rainbow")
        camera_meshes = []
        for i, world_to_camera in enumerate(extrinsics):
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba = colormap(i / max(len(extrinsics), 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            cam_mesh = create_camera_mesh(camera_to_world, color, scene_scale)
            camera_meshes.append(cam_mesh)

        if camera_meshes:
            combined_cam_mesh = trimesh.util.concatenate(camera_meshes)
            scene.add_geometry(combined_cam_mesh, node_name="camera_frustums", geom_name="camera_frustums")

    return apply_scene_alignment(scene, extrinsics)


def _get_filtered_points(
    predictions: dict,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    target_dir: str | None = None,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    conf_thres = max(2.0, float(conf_thres))

    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    images = predictions["images"]

    if mask_sky and target_dir is not None:
        conf = apply_sky_mask(conf, target_dir)

    camera_matrices = predictions["extrinsic"]
    extrinsics = np.zeros((len(camera_matrices), 4, 4), dtype=np.float64)
    extrinsics[:, :3, :4] = camera_matrices
    extrinsics[:, 3, 3] = 1.0

    num_frames = len(points)
    vertices_all = points.reshape(num_frames, -1, 3)
    colors_all = _images_to_rgb(images).reshape(num_frames, -1, 3)
    colors_all = (colors_all * 255).clip(0, 255).astype(np.uint8)
    conf_all = conf.reshape(num_frames, -1)

    flat_vertices = vertices_all.reshape(-1, 3)
    flat_conf = conf_all.reshape(-1)

    mask = np.isfinite(flat_vertices).all(axis=1) & np.isfinite(flat_conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(flat_conf[mask], conf_thres)
        mask &= flat_conf >= conf_threshold
    mask &= flat_conf > 1e-5

    flat_colors = colors_all.reshape(-1, 3)
    if mask_black_bg:
        mask &= flat_colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((flat_colors[:, 0] > 240) & (flat_colors[:, 1] > 240) & (flat_colors[:, 2] > 240))

    mask_per_frame = mask.reshape(num_frames, -1)
    max_points_per_cam = max(1, max_points // num_frames) if max_points > 0 else 0

    per_frame_points = []
    for i in range(num_frames):
        v_i = vertices_all[i][mask_per_frame[i]]
        c_i = colors_all[i][mask_per_frame[i]]
        if len(v_i) > 0:
            cam_to_world = np.linalg.inv(extrinsics[i])
            # Orient all normals towards the camera eye.
            cam_eye = cam_to_world[:3, 3]
            dir_vec = cam_eye[None, :] - v_i
            norms = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
            n_i = (dir_vec / np.maximum(norms, 1e-8)).astype(np.float32)
        else:
            n_i = np.empty((0, 3), dtype=np.float32)

        v_i, c_i, n_i = _limit_points(v_i, c_i, max_points_per_cam, n_i)
        per_frame_points.append((v_i, c_i, n_i))

    return per_frame_points


def save_ply_with_quality(ply_path: str, vertices: np.ndarray, colors: np.ndarray, qualities: np.ndarray):
    """Save vertices, colors, and qualities to a binary PLY file."""
    num_verts = len(vertices)
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {num_verts}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"property float quality\n"
        f"end_header\n"
    )
    
    # Create structured array for fast binary writing
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ('quality', 'f4')
    ]
    data = np.empty(num_verts, dtype=dtype)
    data['x'] = vertices[:, 0]
    data['y'] = vertices[:, 1]
    data['z'] = vertices[:, 2]
    data['r'] = colors[:, 0]
    data['g'] = colors[:, 1]
    data['b'] = colors[:, 2]
    data['quality'] = qualities
    
    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def save_colmap(
    predictions: dict,
    target_dir: str,
    image_names: list[str],
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    max_points: int = 1000000,
) -> str:
    """Export VGGT-Omega predictions to COLMAP format using pycolmap."""
    colmap_dir = os.path.join(target_dir, "colmap")
    if os.path.exists(colmap_dir):
        shutil.rmtree(colmap_dir)
    os.makedirs(colmap_dir, exist_ok=True)

    extrinsics = predictions["extrinsic"]
    intrinsics = predictions["intrinsic"]
    images = predictions["images"]
    points_3d = predictions["world_points_from_depth"]
    depth_conf = predictions["depth_conf"]
    num_frames, height, width, _ = points_3d.shape

    # 1. Apply filtering to get a mask for points
    conf = depth_conf.copy()
    if "depth" in predictions:
        conf[depth_edge(predictions["depth"][..., 0])] = 0.0
    if mask_sky:
        conf = apply_sky_mask(conf, target_dir)

    mask = np.isfinite(points_3d).all(axis=-1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    if mask_black_bg or mask_white_bg:
        colors_rgb = _images_to_rgb(images)
        if mask_black_bg:
            mask &= colors_rgb.sum(axis=-1) >= (16 / 255.0)
        if mask_white_bg:
            mask &= ~((colors_rgb[..., 0] > 0.94) & (colors_rgb[..., 1] > 0.94) & (colors_rgb[..., 2] > 0.94))

    mask = randomly_limit_trues(mask, max_points)

    # 2. Build pycolmap reconstruction
    reconstruction = pycolmap.Reconstruction()

    points_3d_filtered = points_3d[mask]
    colors_rgb_filtered = (_images_to_rgb(images)[mask] * 255).astype(np.uint8)
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    points_xyf_filtered = points_xyf[mask]
    qualities_filtered = conf[mask]

    for i in range(len(points_3d_filtered)):
        reconstruction.add_point3D(points_3d_filtered[i], pycolmap.Track(), colors_rgb_filtered[i])

    # 3. Add cameras and images
    # We'll rescale cameras to original image sizes if possible
    target_size = width # Usually 518
    original_coords = []
    base_image_names = [os.path.basename(n) for n in image_names]

    for img_path in image_names:
        with Image.open(img_path) as img:
            W0, H0 = img.size
            # The model uses load_and_preprocess_images which does:
            # new_width = target_size
            # new_height = round(H0 * (new_width / W0) / 14) * 14
            # then it might crop or pad. 
            # In VGGT-Omega, it usually pads to square or just resizes.
            # Let's assume the intrinsics are already correct for the processed resolution (height, width)
            # and we just need to rescale to (H0, W0).
            
            # Simple rescaling:
            s_x = W0 / width
            s_y = H0 / height
            original_coords.append([0, 0, W0, H0, s_x, s_y])

    for fidx in range(num_frames):
        # Camera
        pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, "PINHOLE")
        W0, H0 = int(original_coords[fidx][2]), int(original_coords[fidx][3])
        s_x, s_y = original_coords[fidx][4], original_coords[fidx][5]
        
        # Rescale intrinsic parameters
        pycolmap_intri[0] *= s_x # fx
        pycolmap_intri[1] *= s_y # fy
        pycolmap_intri[2] *= s_x # cx
        pycolmap_intri[3] *= s_y # cy

        camera = pycolmap.Camera(
            model="PINHOLE", width=W0, height=H0, params=pycolmap_intri, camera_id=fidx + 1
        )
        reconstruction.add_camera(camera)

        # Image
        # E is W2C: [R | t]
        E = extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(E[:3, :3]), E[:3, 3])
        image = pycolmap.Image(
            id=fidx + 1, name=base_image_names[fidx], camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        # 4. Add point correspondences (tracks)
        points2D_list = []
        point2D_idx = 0
        
        # Find points belonging to this frame
        frame_mask = points_xyf_filtered[:, 2].astype(np.int32) == fidx
        frame_point_indices = np.nonzero(frame_mask)[0]
        
        for p_idx in frame_point_indices:
            point3D_id = p_idx + 1
            xy = points_xyf_filtered[p_idx, :2] * [s_x, s_y]
            
            points2D_list.append(pycolmap.Point2D(xy, point3D_id))
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1
        
        image.points2D = pycolmap.ListPoint2D(points2D_list)
        image.registered = True
        reconstruction.add_image(image)

    # 5. Compute and apply the same scene alignment used by the GLB exporter
    #    so that COLMAP/PLY and GLB share the same coordinate frame in MeshLab.
    T_align = get_scene_alignment_transform(extrinsics)
    T_align_inv = np.linalg.inv(T_align)

    # Transform 3D points: apply T_align to each point
    pts_h = np.concatenate([points_3d_filtered, np.ones((len(points_3d_filtered), 1))], axis=1)
    points_3d_aligned = (T_align @ pts_h.T).T[:, :3]

    # Update point3D positions in the reconstruction
    for idx, pt3d_id in enumerate(reconstruction.point3D_ids()):
        reconstruction.points3D[pt3d_id].xyz = points_3d_aligned[idx]

    # Transform extrinsics: new_E = old_E @ T_align_inv (world-to-camera in new world frame)
    for fidx in range(num_frames):
        img_id = fidx + 1
        old_img = reconstruction.images[img_id]
        E4 = np.eye(4)
        E4[:3, :3] = old_img.cam_from_world.rotation.matrix()
        E4[:3, 3] = old_img.cam_from_world.translation
        E4_new = E4 @ T_align_inv
        new_cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(E4_new[:3, :3]), E4_new[:3, 3])
        old_img.cam_from_world = new_cam_from_world
        reconstruction.images[img_id] = old_img

    # 6. Write reconstruction and PLY file
    reconstruction.write(colmap_dir)

    # ply_path = os.path.join(colmap_dir, "points_with_quality.ply")
    # save_ply_with_quality(ply_path, points_3d_aligned, colors_rgb_filtered, qualities_filtered)

    return colmap_dir


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(
    vertices: np.ndarray, colors: np.ndarray, max_points: int, normals: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors, normals
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    norm_res = normals[indices] if normals is not None else None
    return vertices[indices], colors[indices], norm_res


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def create_camera_mesh(transform: np.ndarray, face_colors: tuple, scene_scale: float) -> trimesh.Trimesh:
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    complete_transform = transform @ get_opengl_conversion_matrix() @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices = transform_points(complete_transform, vertices)

    camera_mesh = trimesh.Trimesh(vertices=vertices, faces=compute_camera_faces(camera_cone_shape))
    camera_mesh.visual.face_colors[:, :3] = face_colors
    return camera_mesh


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    camera_mesh = create_camera_mesh(transform, face_colors, scene_scale)
    scene.add_geometry(camera_mesh)


def get_scene_alignment_transform(extrinsics: np.ndarray) -> np.ndarray:
    """Return the 4x4 world-space alignment matrix used when exporting to GLB.

    This transforms raw VGGT-Omega world coordinates into the final scene frame:
    1. Transform into camera-0's frame: inv(extrinsics[0])
    2. Convert OpenCV -> OpenGL convention (flip Y and Z): opengl_conversion_matrix
    3. Rotate 180° around Y to flip Z forward: align_rotation
    """
    opengl_conversion_matrix = get_opengl_conversion_matrix()
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Support both (3, 4) raw extrinsics and (4, 4) augmented extrinsics
    E0 = np.eye(4, dtype=np.float64)
    E0[:3, :4] = extrinsics[0][:3, :4]
    return np.linalg.inv(E0) @ opengl_conversion_matrix @ align_rotation


def apply_scene_alignment(scene: trimesh.Scene, extrinsics: np.ndarray) -> trimesh.Scene:
    scene.apply_transform(get_scene_alignment_transform(extrinsics))
    return scene


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int | None = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces = []
    num_vertices = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices

        faces.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces += [(v3, v2, v1) for v1, v2, v3 in faces]
    return np.array(faces)


def apply_sky_mask(conf: np.ndarray, target_dir: str) -> np.ndarray:
    print("\n--- Starting Sky Masking (ONNX Segment Sky) ---")
    t_start_sky_mask = time.time()

    image_dir = os.path.join(target_dir, "images")
    image_names = sorted(os.listdir(image_dir))
    height, width = conf.shape[-2:]
    masks = []
    skyseg_session = None

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(target_dir, "sky_masks", image_name)
        if os.path.exists(mask_path):
            sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            if not os.path.exists("skyseg.onnx"):
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                    "skyseg.onnx",
                )
            if skyseg_session is None:
                import onnxruntime
                t0 = time.time()
                skyseg_session = onnxruntime.InferenceSession("skyseg.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                t1 = time.time()
                providers = skyseg_session.get_providers()
                device_type = "GPU" if any("GPU" in p or "CUDA" in p or "ROCM" in p or "DirectML" in p for p in providers) else "CPU"
                print(f"[Stage: Sky Mask] Initialized ONNX InferenceSession in {t1 - t0:.4f} seconds (Running on: {device_type}, Providers: {providers})")
            
            print(f"[Stage: Sky Mask] Segmenting sky for {image_name}...")
            t_seg_start = time.time()
            sky_mask = segment_sky(image_path, skyseg_session, mask_path)
            t_seg_end = time.time()
            print(f"[Stage: Sky Mask] Segmented sky for {image_name} in {t_seg_end - t_seg_start:.4f} seconds.")

        if sky_mask.shape != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height))
        masks.append(sky_mask)

    t_end_sky_mask = time.time()
    print(f"--- Sky Masking completed in {t_end_sky_mask - t_start_sky_mask:.4f} seconds ---\n")

    return conf * (np.array(masks) > 0.1).astype(np.float32)


def segment_sky(image_path: str, onnx_session, mask_filename: str) -> np.ndarray:
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map)
    output_mask[result_map < 32] = 255

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size: list[int], image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")


def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    """
    If mask has more than max_trues True values,
    randomly keep only max_trues of them and set the rest to False.
    """
    true_indices = np.flatnonzero(mask)
    if true_indices.size <= max_trues:
        return mask
    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)
    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True
    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames: int, height: int, width: int) -> np.ndarray:
    """Creates a grid of pixel coordinates and frame indices."""
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    return np.stack((x_coords, y_coords, f_coords), axis=-1)


def _build_pycolmap_intri(fidx: int, intrinsics: np.ndarray, camera_type: str) -> np.ndarray:
    """Helper to get camera parameters for pycolmap."""
    if camera_type == "PINHOLE":
        return np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        return np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    return np.array([])
