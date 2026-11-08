import os
import time
import open3d as o3d
import numpy as np

from scipy.ndimage import median_filter
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp


def draw_camera(c2w, cam_width=0.24/2, cam_height=0.16/2, f=0.10, color=[0, 1, 0], show_axis=True):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        return [line_set, axis]
    else:
        return [line_set]


def interactive_pcd_viewer(pts_raw, colors_raw, conf_raw,
                           pts_raw_0=None, colors_raw_0=None, conf_raw_0=None,
                           c2w=None, c2w_0=None,
                           images=None, pts_full=None, conf_full=None,
                           initial_conf_thresh=0.0,
                           edge_normal_threshold=5.0,
                           edge_depth_threshold=0.008):
    """
    Interactive Open3D viewer with keyboard controls:
      [Right]/[Left] : increase/decrease confidence threshold by 0.01 (Right=Increase, Left=Decrease)
      [Up]/[Down]  : increase/decrease confidence threshold by 0.1
      [+]/[-]      : increase/decrease point size
      [B]          : toggle background color (Black/White)
      [E]          : toggle edge mask (get_pts_mask)
      [0]/[1]      : switch between pcd0 (initial) and pcd (refined)
      [Space]      : print current camera parameters
      [Q]/[Esc]    : close

    Args:
        pts_raw: (N, 3) flattened world points
        colors_raw: (N, 3) flattened RGB colors
        conf_raw: (N,) confidence values (sigmoid-transformed)
        pts_raw_0 / colors_raw_0 / conf_raw_0: optional alternative point cloud
        c2w: (T, 4, 4) or list of poses for refined trajectory (pcd)
        c2w_0: (T, 4, 4) or list of poses for initial trajectory (pcd0)
        images: (B, T, C, H, W) tensor for sky mask (for get_pts_mask)
        pts_full: (B*T, H, W, 3) array for edge detection (for get_pts_mask)
        conf_full: (B*T, H, W) array for edge detection (for get_pts_mask)
        initial_conf_thresh: initial confidence threshold
        edge_normal_threshold: angle tolerance (deg)
        edge_depth_threshold: relative depth tolerance
    """

    # --- State ---
    class State:
        conf_thresh = initial_conf_thresh
        edge_mask_enabled = False
        showing_pcd0 = False
        edge_mask_flat = None       # for pcd
        edge_mask_flat_0 = None     # for pcd0
        point_size = 1.0
        bg_black = True
        camera_geometries = []      # list of current camera geometries

    state = State()
    pcd_handle = o3d.geometry.PointCloud()

    def _get_camera_geometries(poses, color):
        geoms = []
        if poses is None:
            return geoms
        # Handle (B, T, 4, 4) or (T, 4, 4)
        if hasattr(poses, 'shape') and len(poses.shape) == 4:
            poses = poses.reshape(-1, 4, 4)
        elif hasattr(poses, 'ndim') and poses.ndim == 4: # numpy
            poses = poses.reshape(-1, 4, 4)

        for pose in poses:
            geoms.extend(draw_camera(pose, color=color))
        return geoms

    def _build_pcd():
        """Rebuild point cloud geometry from current state."""
        if state.showing_pcd0 and pts_raw_0 is not None:
            pts, colors, conf = pts_raw_0, colors_raw_0, conf_raw_0
            edge_mask = state.edge_mask_flat_0
        else:
            pts, colors, conf = pts_raw, colors_raw, conf_raw
            edge_mask = state.edge_mask_flat

        mask = conf >= state.conf_thresh
        if state.edge_mask_enabled and edge_mask is not None:
            mask = mask & edge_mask

        if mask.sum() == 0:
            pcd_handle.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            pcd_handle.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        else:
            pcd_handle.points = o3d.utility.Vector3dVector(pts[mask].reshape(-1, 3))
            pcd_handle.colors = o3d.utility.Vector3dVector(colors[mask].reshape(-1, 3))

        n = len(pcd_handle.points)
        src = "pcd0" if state.showing_pcd0 else "pcd"
        edge_str = " [edge mask ON]" if state.edge_mask_enabled else ""
        print(f"  conf_thresh={state.conf_thresh:.4f} | {src} | {n:,} points{edge_str} | pt_size={state.point_size:.1f}")

    def _refresh(vis):
        _build_pcd()
        vis.update_geometry(pcd_handle)

        opt = vis.get_render_option()
        opt.point_size = state.point_size
        opt.background_color = np.array([0, 0, 0]) if state.bg_black else np.array([1, 1, 1])

        vis.update_renderer()

    def _update_camera_vis(vis):
        """Update visualized cameras based on current source (pcd0 vs pcd)."""
        # Remove old
        for g in state.camera_geometries:
            vis.remove_geometry(g, reset_bounding_box=False)

        # Determine new poses
        if state.showing_pcd0:
            poses = c2w_0
            color = [1, 0, 0] # Red for initial
        else:
            poses = c2w
            color = [1, 0, 0] # Green for refined

        # Add new
        new_geoms = _get_camera_geometries(poses, color)
        for g in new_geoms:
            vis.add_geometry(g, reset_bounding_box=False)
        state.camera_geometries = new_geoms

    def _compute_edge_mask():
        if pts_full is None or conf_full is None:
            print("Warning: pts_full and conf_full required for edge mask.")
            return

        from amb3r.tools.pts_vis import get_pts_mask
        print("Computing edge mask...")
        mask, _ = get_pts_mask(
            pts_full,
            images=images,
            conf=conf_full,
            conf_threshold=state.conf_thresh,
            edge_normal_threshold=edge_normal_threshold,
            edge_depth_threshold=edge_depth_threshold
        )
        state.edge_mask_flat = mask.reshape(-1)
        print("Edge mask computed.")

    # --- Key Callbacks ---
    def thresh_up_small(vis):
        state.conf_thresh = min(1.0, state.conf_thresh + 0.01)
        _refresh(vis)
        return False

    def thresh_down_small(vis):
        state.conf_thresh = max(0.0, state.conf_thresh - 0.01)
        _refresh(vis)
        return False

    def thresh_up_large(vis):
        state.conf_thresh = min(1.0, state.conf_thresh + 0.1)
        _refresh(vis)
        return False

    def thresh_down_large(vis):
        state.conf_thresh = max(0.0, state.conf_thresh - 0.1)
        _refresh(vis)
        return False

    def toggle_edge_mask(vis):
        state.edge_mask_enabled = not state.edge_mask_enabled
        if state.edge_mask_enabled and state.edge_mask_flat is None:
            _compute_edge_mask()
        _refresh(vis)
        return False

    def switch_to_pcd0(vis):
        if pts_raw_0 is not None:
            state.showing_pcd0 = True
            _refresh(vis)
            _update_camera_vis(vis)
        return False

    def switch_to_pcd1(vis):
        state.showing_pcd0 = False
        _refresh(vis)
        _update_camera_vis(vis)
        return False

    def pt_size_up(vis):
        state.point_size += 1.0
        _refresh(vis)
        return False

    def pt_size_down(vis):
        state.point_size = max(1.0, state.point_size - 1.0)
        _refresh(vis)
        return False

    def toggle_bg(vis):
        state.bg_black = not state.bg_black
        _refresh(vis)
        return False

    def print_camera(vis):
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        print("\nCamera intrinsics:")
        print(params.intrinsic.intrinsic_matrix)
        print("Camera extrinsics:")
        print(params.extrinsic)
        return False

    # --- Build initial point cloud ---
    _build_pcd()

    # --- Visualizer setup ---
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # Show controls in the window title since we can't do 2D overlays easily in legacy mode
    instructions = "Controls: [Right/Left] Conf +/-0.01 | [Up/Down] Conf +/-0.1 | [+/-] Pt Size | [B] BG Color | [E] Edge Mask | [0/1] Switch PCD"
    vis.create_window(window_name=instructions, width=1920, height=1080)
    vis.add_geometry(pcd_handle)

    # Initial camera geometries
    _update_camera_vis(vis)

    opt = vis.get_render_option()
    opt.point_size = state.point_size
    opt.background_color = np.array([0, 0, 0])

    # Register key callbacks (ASCII codes)
    vis.register_key_callback(263, thresh_up_small)      # Right arrow
    vis.register_key_callback(262, thresh_down_small)    # Left arrow
    vis.register_key_callback(265, thresh_up_large)      # Up arrow
    vis.register_key_callback(264, thresh_down_large)    # Down arrow

    vis.register_key_callback(ord('='), pt_size_up)      # +/= key
    vis.register_key_callback(ord('+'), pt_size_up)      # + (numpad or shift)
    vis.register_key_callback(ord('-'), pt_size_down)    # -/- key
    vis.register_key_callback(ord('_'), pt_size_down)    # _ (shift -)

    vis.register_key_callback(ord('B'), toggle_bg)       # B key
    vis.register_key_callback(ord('b'), toggle_bg)       # b key

    vis.register_key_callback(ord('E'), toggle_edge_mask)    # E key
    vis.register_key_callback(ord('e'), toggle_edge_mask)    # e key

    vis.register_key_callback(ord('0'), switch_to_pcd0)      # 0 key
    vis.register_key_callback(ord('1'), switch_to_pcd1)      # 1 key
    vis.register_key_callback(32, print_camera)              # Spacebar

    print("\n--- Interactive PCD Viewer Controls ---")
    print("  [Right/Left]: confidence threshold ±0.01 (Right=+, Left=-)")
    print("  [Up/Down]   : confidence threshold ±0.1")
    print("  [+/-]       : point size increase/decrease")
    print("  [B]         : toggle background color (Black/White)")
    print("  [E]         : toggle edge mask (get_pts_mask)")
    print("  [0/1]       : switch pcd0 (initial) / pcd (refined)")
    print("  [Space]     : print camera parameters")
    print("  [Q / Esc]   : close")
    print("---------------------------------------\n")

    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Camera frustum helpers
# ---------------------------------------------------------------------------

def create_camera_frustum_pcd(cam_width, cam_height, f, extrinsic, color_val, points_per_line=40):
    """Camera frustum as a PointCloud (interpolated points along edges)."""
    points = np.array([
        [0, 0, 0],
        [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
        [cam_width, cam_height, f],   [-cam_width, cam_height, f],
    ])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    t_vals = np.linspace(0., 1., points_per_line)
    all_pts = []
    for a, b in lines:
        all_pts.append(points[a] * (1 - t_vals)[:, None] + points[b] * t_vals[:, None])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_pts))
    pcd.paint_uniform_color(color_val)
    pcd.transform(extrinsic)
    return pcd


def create_sphere_pcd_template(radius, resolution=8):
    """Unit sphere PointCloud template used to give frustum edges thickness."""
    phi, theta = np.meshgrid(
        np.linspace(0, np.pi, resolution),
        np.linspace(0, 2 * np.pi, resolution * 2),
    )
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))
    return pcd


def create_thick_frustum_pcd(cam_width, cam_height, f, extrinsic, color_val,
                              sphere_template, points_per_line=10):
    """Camera frustum PointCloud with thickness via sphere-template convolution."""
    points = np.array([
        [0, 0, 0],
        [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
        [cam_width, cam_height, f],   [-cam_width, cam_height, f],
    ])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    t_vals = np.linspace(0., 1., points_per_line)
    all_pts = []
    for a, b in lines:
        all_pts.append(points[a] * (1 - t_vals)[:, None] + points[b] * t_vals[:, None])
    base_pts = np.concatenate(all_pts)
    tmpl_pts = np.asarray(sphere_template.points)
    final_pts = (base_pts[:, None, :] + tmpl_pts[None, :, :]).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.paint_uniform_color(color_val)
    pcd.transform(extrinsic)
    return pcd


def create_trajectory_pcd(positions, color_val, points_per_segment=20):
    """Trajectory PointCloud interpolated between consecutive camera positions."""
    t_vals = np.linspace(0., 1., points_per_segment)
    all_pts = []
    for i in range(len(positions) - 1):
        s, e = positions[i], positions[i + 1]
        all_pts.append(s * (1 - t_vals)[:, None] + e * t_vals[:, None])
    if not all_pts:
        return o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_pts))
    pcd.paint_uniform_color(color_val)
    return pcd


# ---------------------------------------------------------------------------
# Trajectory smoothing & interpolation  (requires scipy)
# ---------------------------------------------------------------------------

def smooth_trajectory_robust(poses, translation_smoothing_factor=1.0, rotation_filter_window=3):
    """Smooth a (N, 4, 4) pose trajectory, preserving exact start/end points."""
    num_poses = len(poses)
    if num_poses < 3:
        return poses

    first_pose, last_pose = poses[0].copy(), poses[-1].copy()
    t = np.arange(num_poses)

    print(f"Smoothing translation (s={translation_smoothing_factor})...")
    positions = poses[:, :3, 3]
    try:
        k_val = min(3, num_poses - 1)
        tck, _ = splprep(positions.T, u=t, k=k_val, s=translation_smoothing_factor)
        pos_smooth = np.array(splev(np.linspace(t[0], t[-1], num_poses), tck)).T
    except Exception as e:
        print(f"WARNING: Spline fitting failed ({e}). Using median filter.")
        pos_smooth = median_filter(positions, size=(rotation_filter_window, 1), mode='reflect')

    print(f"Smoothing rotation (window={rotation_filter_window})...")
    if rotation_filter_window % 2 == 0:
        rotation_filter_window += 1
    if num_poses <= rotation_filter_window:
        rotation_filter_window = max(1, (num_poses // 2) * 2 - 1) or 1

    try:
        rotations = Rotation.from_matrix(poses[:, :3, :3])
    except ValueError as e:
        print(f"ERROR: Invalid rotation matrices. {e}")
        return poses

    quats = rotations.as_quat()
    for i in range(1, num_poses):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1

    quats_smooth = median_filter(quats, size=(rotation_filter_window, 1), mode='reflect')
    norms = np.linalg.norm(quats_smooth, axis=1)
    norms[norms == 0] = 1.0
    quats_smooth /= norms[:, None]
    rot_smooth = Rotation.from_quat(quats_smooth).as_matrix()

    poses_smooth = np.eye(4)[None].repeat(num_poses, axis=0)
    poses_smooth[:, :3, :3] = rot_smooth
    poses_smooth[:, :3, 3] = pos_smooth

    print("  ... Forcing path to match exact start and end points.")
    poses_smooth[0] = first_pose
    poses_smooth[-1] = last_pose
    return poses_smooth


def interpolate_smooth_trajectory(poses_smooth, num_output_frames):
    """Interpolate a smooth (N, 4, 4) path to num_output_frames via spline + slerp."""
    print(f"Interpolating {len(poses_smooth)} poses to {num_output_frames} frames...")
    positions = poses_smooth[:, :3, 3]
    rotations = Rotation.from_matrix(poses_smooth[:, :3, :3])
    t_in = np.linspace(0, 1, len(poses_smooth))
    t_out = np.linspace(0, 1, num_output_frames)

    k_val = min(3, len(poses_smooth) - 1)
    tck, _ = splprep(positions.T, u=t_in, k=k_val, s=0)
    pos_interp = np.array(splev(t_out, tck)).T

    rot_interp = Slerp(t_in, rotations)(t_out).as_matrix()

    poses_interp = np.eye(4)[None].repeat(num_output_frames, axis=0)
    poses_interp[:, :3, :3] = rot_interp
    poses_interp[:, :3, 3] = pos_interp
    return poses_interp


# ---------------------------------------------------------------------------
# Scene building
# ---------------------------------------------------------------------------

def convert_pinhole_params_to_poses(param_list):
    """Convert a list of PinholeCameraParameters to (N, 4, 4) c2w poses."""
    return np.array([np.linalg.inv(p.extrinsic) for p in param_list])


def build_full_scene_geometries(
    pts_all, image_all, poses_all, kf_idx_all, mask,
    cam_width, cam_height, focal_length,
    frustum_color, frustum_sphere_template,
    points_per_line=100,
):
    """Build full-resolution PCD (keyframes only) + all-frame frustum PCD."""
    print("Building FULL-RESOLUTION scene (point cloud + all frustums)...")
    t = pts_all.shape[0]
    pcd = o3d.geometry.PointCloud()
    all_frustums_pcd = o3d.geometry.PointCloud()
    kf_set = set(kf_idx_all)

    for i in range(t):
        if i in kf_set:
            new_pts = pts_all[i].reshape(-1, 3)
            new_colors = image_all[i].reshape(-1, 3)
            if mask is not None:
                m = mask[i].reshape(-1).astype(bool)
                new_pts, new_colors = new_pts[m], new_colors[m]
            if new_pts.shape[0] > 0:
                pcd.points.extend(o3d.utility.Vector3dVector(new_pts))
                pcd.colors.extend(o3d.utility.Vector3dVector(new_colors))

        frustum = create_thick_frustum_pcd(
            cam_width, cam_height, focal_length, poses_all[i],
            frustum_color, frustum_sphere_template, points_per_line,
        )
        all_frustums_pcd.points.extend(frustum.points)
        all_frustums_pcd.colors.extend(frustum.colors)
        if (i + 1) % 100 == 0 or i == t - 1:
            print(f"  ... Processed frame {i+1}/{t}")

    return pcd, all_frustums_pcd


def build_frustums_pcd(
    poses_all, cam_width, cam_height, focal_length,
    frustum_color, frustum_sphere_template, points_per_line=10,
):
    """Build a single PointCloud of all camera frustums from poses_all."""
    all_frustums_pcd = o3d.geometry.PointCloud()
    for pose in poses_all:
        frustum = create_thick_frustum_pcd(
            cam_width, cam_height, focal_length, pose,
            frustum_color, frustum_sphere_template, points_per_line,
        )
        all_frustums_pcd.points.extend(frustum.points)
        all_frustums_pcd.colors.extend(frustum.colors)
    return all_frustums_pcd


def build_interactive_scene(
    pts_all, image_all, poses_all, kf_idx_all, mask,
    cam_width, cam_height, focal_length,
    frustum_color, frustum_sphere_template,
    points_per_line=10, max_points=2_000_000,
    pcd_scene=None,
):
    """Build downsampled PCD + all-frame frustum PCD for interactive use.

    If pcd_scene is provided it is used directly (and downsampled if needed),
    skipping the per-frame rebuild from pts_all.
    """
    print(f"Building FAST interactive scene (max {max_points} points)...")

    all_frustums_pcd = build_frustums_pcd(
        poses_all, cam_width, cam_height, focal_length,
        frustum_color, frustum_sphere_template, points_per_line,
    )
    print("  ... All frustums built.")

    pcd_interactive = o3d.geometry.PointCloud()

    if pcd_scene is not None:
        pts_cat = np.asarray(pcd_scene.points)
        colors_cat = np.asarray(pcd_scene.colors)
    else:
        kf_set = set(kf_idx_all)
        all_pts_list, all_colors_list = [], []
        for i in kf_set:
            new_pts = pts_all[i].reshape(-1, 3)
            new_colors = image_all[i].reshape(-1, 3)
            if mask is not None:
                m = mask[i].reshape(-1).astype(bool)
                new_pts, new_colors = new_pts[m], new_colors[m]
            if new_pts.shape[0] > 0:
                all_pts_list.append(new_pts)
                all_colors_list.append(new_colors)
        if not all_pts_list:
            return pcd_interactive, all_frustums_pcd
        pts_cat = np.concatenate(all_pts_list)
        colors_cat = np.concatenate(all_colors_list)

    n = len(pts_cat)
    if n > max_points:
        print(f"  Original scene has {n} points. Downsampling to {max_points}...")
        idx = np.random.choice(n, size=max_points, replace=False)
        pts_cat, colors_cat = pts_cat[idx], colors_cat[idx]
    else:
        print(f"  Scene has {n} points (<= {max_points}). Using all.")

    pcd_interactive.points = o3d.utility.Vector3dVector(pts_cat)
    pcd_interactive.colors = o3d.utility.Vector3dVector(colors_cat)
    print("  ... Interactive point cloud built.")
    return pcd_interactive, all_frustums_pcd


# ---------------------------------------------------------------------------
# Static viewer
# ---------------------------------------------------------------------------

def draw_scene(geometries, point_size=1.0, window_name="Open3D", width=1920, height=1080):
    """Static viewer for a list of geometries with configurable point size."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Interactive camera path collection & offline video rendering
# ---------------------------------------------------------------------------

def run_path_preview(pcd, all_frustums, saved_frustums, interpolated_poses, fps=60, point_size=1.0):
    """Play a fly-through preview of interpolated_poses in a temporary window."""
    P_WIDTH, P_HEIGHT = 1280, 720
    vis_prev = o3d.visualization.Visualizer()
    vis_prev.create_window(width=P_WIDTH, height=P_HEIGHT, visible=True)

    if not pcd.is_empty():
        vis_prev.add_geometry(pcd)
    if not all_frustums.is_empty():
        vis_prev.add_geometry(all_frustums)
    for f in saved_frustums:
        vis_prev.add_geometry(f)

    opt = vis_prev.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0, 0, 0])
    opt.show_coordinate_frame = True

    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        P_WIDTH, P_HEIGHT, 800, 800, P_WIDTH / 2, P_HEIGHT / 2
    )
    ctr = vis_prev.get_view_control()
    frame_dur = 1.0 / fps

    for pose in interpolated_poses:
        t0 = time.time()
        cam_params.extrinsic = np.linalg.inv(pose)
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        vis_prev.poll_events()
        vis_prev.update_renderer()
        dt = time.time() - t0
        if frame_dur - dt > 0:
            time.sleep(frame_dur - dt)

    vis_prev.destroy_window()


def collect_manual_camera_path(
    pcd_scene, all_frustums_scene,
    cam_width, cam_height, focal_length,
    frustum_sphere_template, selected_cam_color,
    num_video_frames, preview_s_factor, preview_rot_window,
    initial_poses=None, point_size=1.0,
):
    """Open an interactive window to place camera keyframes for video rendering.

    Returns:
        (saved_params, is_loop): list of PinholeCameraParameters and the final loop flag.

    Controls:
        [C] Save current view as keyframe
        [Z] Undo last keyframe
        [P] Play preview
        [L] Toggle loop (appends first keyframe to end)
        [Q] Finish and quit
        [A] Toggle coordinate axis
    """
    print("\n" + "=" * 50)
    print("Starting Interactive Camera Placement")
    print("   Scene is downsampled for performance.")
    print("\n  --- MOUSE CONTROLS ---")
    print("  - Orbit (Rotate View):  Left-Click + Drag")
    print("  - Pan (Move Camera):    Shift + Left-Click + Drag (or Ctrl+Click)")
    print("  - Zoom (Move In/Out):   Scroll Wheel")
    print("\n  --- KEY CONTROLS ---")
    print("  - Save View:            Press [C]")
    print("  - Undo Last View:       Press [Z]")
    print("  - Play Full Preview:    Press [P]")
    print("  - Toggle Loop:          Press [L]  (currently: OFF)")
    print("  - Finish and Quit:      Press [Q]")
    print("  - Show/Hide View Axis:  Press [A]")
    print("=" * 50)

    W_WIDTH, W_HEIGHT = 1920, 1080
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=W_WIDTH, height=W_HEIGHT, visible=True)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0, 0, 0])
    opt.show_coordinate_frame = True

    saved_params = []
    saved_frustums = []
    traj_vis = [None]   # mutable container to allow update inside closures
    loop_state = [False]

    if initial_poses is not None and len(initial_poses) > 0:
        print(f"  ... Loading {len(initial_poses)} existing camera keyframes.")
        dummy_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        for pose in initial_poses:
            p = o3d.camera.PinholeCameraParameters()
            p.intrinsic = dummy_params.intrinsic
            p.extrinsic = np.linalg.inv(pose)
            frustum = create_thick_frustum_pcd(
                cam_width, cam_height, focal_length, pose,
                selected_cam_color, frustum_sphere_template, 40,
            )
            saved_params.append(p)
            saved_frustums.append(frustum)
            vis.add_geometry(frustum, reset_bounding_box=False)

    if not all_frustums_scene.is_empty():
        vis.add_geometry(all_frustums_scene)
    if not pcd_scene.is_empty():
        vis.add_geometry(pcd_scene)

    def _update_traj(vis):
        if traj_vis[0] is not None:
            vis.remove_geometry(traj_vis[0], reset_bounding_box=False)
            traj_vis[0] = None
        if len(saved_params) < 2:
            return
        manual_poses = convert_pinhole_params_to_poses(saved_params)
        if loop_state[0]:
            manual_poses = np.concatenate([manual_poses, manual_poses[:1]])
        poses_smooth = smooth_trajectory_robust(manual_poses, preview_s_factor, preview_rot_window)
        interp = interpolate_smooth_trajectory(poses_smooth, num_video_frames)
        pts = interp[:, :3, 3]
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.paint_uniform_color([0.2, 0.8, 0.2])
        vis.add_geometry(line_set, reset_bounding_box=False)
        traj_vis[0] = line_set

    _update_traj(vis)
    vis.update_renderer()

    def save_camera_cb(v):
        params = v.get_view_control().convert_to_pinhole_camera_parameters()
        pose = np.linalg.inv(params.extrinsic)
        frustum = create_thick_frustum_pcd(
            cam_width, cam_height, focal_length, pose,
            selected_cam_color, frustum_sphere_template, 40,
        )
        saved_params.append(params)
        saved_frustums.append(frustum)
        v.add_geometry(frustum, reset_bounding_box=False)
        _update_traj(v)
        v.update_renderer()
        print(f"  ... Camera keyframe #{len(saved_params)} SAVED!")
        return True

    def revert_camera_cb(v):
        if not saved_params:
            print("  ... No keyframes to revert.")
            return True
        v.remove_geometry(saved_frustums.pop(), reset_bounding_box=False)
        saved_params.pop()
        _update_traj(v)
        v.update_renderer()
        print(f"  ... Reverted camera. {len(saved_params)} keyframes remaining.")
        return True

    def preview_cb(v):
        if len(saved_params) < 2:
            print("  ... Need at least 2 keyframes to play a preview.")
            return True
        print(f"  ... Smoothing and playing {num_video_frames}-frame preview...")
        manual_poses = convert_pinhole_params_to_poses(saved_params)
        if loop_state[0]:
            manual_poses = np.concatenate([manual_poses, manual_poses[:1]])
        poses_smooth = smooth_trajectory_robust(manual_poses, preview_s_factor, preview_rot_window)
        interp = interpolate_smooth_trajectory(poses_smooth, num_video_frames)
        run_path_preview(pcd_scene, all_frustums_scene, saved_frustums, interp, point_size=point_size)
        print("  ... Preview finished. Returning to editor.")
        return True

    def toggle_loop_cb(v):
        loop_state[0] = not loop_state[0]
        print(f"  ... Loop {'ON' if loop_state[0] else 'OFF'}")
        _update_traj(v)
        v.update_renderer()
        return True

    def quit_cb(_v):
        print("  ... Path collection finished. Closing window.")
        vis.destroy_window()
        return True

    def toggle_axis_cb(v):
        opt = v.get_render_option()
        opt.show_coordinate_frame = not opt.show_coordinate_frame
        v.update_renderer()
        return True

    vis.register_key_callback(ord("C"), save_camera_cb)
    vis.register_key_callback(ord("Z"), revert_camera_cb)
    vis.register_key_callback(ord("P"), preview_cb)
    vis.register_key_callback(ord("L"), toggle_loop_cb)
    vis.register_key_callback(ord("Q"), quit_cb)
    vis.register_key_callback(ord("A"), toggle_axis_cb)

    vis.run()
    return saved_params, loop_state[0]


def render_offline_video(
    pts_all, image_all, poses_all, kf_idx_all, mask,
    interpolated_poses, output_dir,
    cam_width, cam_height, focal_length,
    frustum_color, frustum_sphere_template, save_video=True,
    pcd_scene=None, point_size=1.0,
):
    """Render an offline fly-through video from interpolated_poses and save to output_dir.

    If pcd_scene is provided it is used directly, skipping the per-frame rebuild from
    pts_all (much faster when you already have a pre-built point cloud).
    """
    import imageio

    if pcd_scene is None:
        pcd_scene, all_frustums_scene = build_full_scene_geometries(
            pts_all, image_all, poses_all, kf_idx_all, mask,
            cam_width, cam_height, focal_length,
            frustum_color, frustum_sphere_template, points_per_line=100,
        )
    else:
        print("Using provided point cloud; building frustums only...")
        all_frustums_scene = build_frustums_pcd(
            poses_all, cam_width, cam_height, focal_length,
            frustum_color, frustum_sphere_template, points_per_line=100,
        )

    print(f"\nStarting offline render of {len(interpolated_poses)} frames...")

    W_WIDTH, W_HEIGHT = 1920, 1080
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W_WIDTH, height=W_HEIGHT, visible=True)

    render_frame_path = os.path.join(output_dir, 'render_frames_chase_cam')
    os.makedirs(render_frame_path, exist_ok=True)
    video_path = os.path.join(output_dir, 'render_chase_cam.mp4')

    writer = imageio.get_writer(video_path, fps=30) if save_video else None

    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        W_WIDTH, W_HEIGHT, 1200, 1200, W_WIDTH / 2, W_HEIGHT / 2
    )

    if not pcd_scene.is_empty():
        vis.add_geometry(pcd_scene)
    if not all_frustums_scene.is_empty():
        vis.add_geometry(all_frustums_scene)

    CURRENT_CAM_COLOR = [0.2, 0.2, 0.8]

    try:
        for i, K_i in enumerate(interpolated_poses):
            cam_params.extrinsic = np.linalg.inv(K_i)
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

            opt = vis.get_render_option()
            opt.point_size = point_size
            opt.background_color = np.array([0, 0, 0])

            vis.poll_events()
            vis.update_renderer()

            frame = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(render_frame_path, f'frame_{i:05d}.png'), frame)
            if writer is not None:
                writer.append_data(frame)

            # Dummy add/remove to trigger a scene refresh on next iteration
            dummy = create_thick_frustum_pcd(
                cam_width, cam_height, focal_length, K_i, CURRENT_CAM_COLOR,
                frustum_sphere_template, 1,
            )
            vis.add_geometry(dummy, reset_bounding_box=False)
            vis.remove_geometry(dummy, reset_bounding_box=False)

            if (i + 1) % 50 == 0 or i == len(interpolated_poses) - 1:
                print(f"  ... Rendered frame {i+1}/{len(interpolated_poses)}")
    finally:
        if writer is not None:
            writer.close()
        vis.destroy_window()
