import time
import numpy as np
import openvr
import open3d as o3d


# -------------------- OpenVR helpers --------------------

def _mat34_to_Rp(mat34):
    M = np.array(list(mat34), dtype=np.float64).reshape(3, 4)
    return M[:, :3].copy(), M[:, 3].copy()

def _T_from_Rp(R, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def _list_controllers(vrsys):
    out = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vrsys.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
            continue
        role = vrsys.getControllerRoleForTrackedDeviceIndex(i)
        out.append((i, role))
    return out

def _pick_controller(vrsys, prefer="right"):
    ctrls = _list_controllers(vrsys)
    if not ctrls:
        return None, ctrls
    prefer = prefer.lower()
    if prefer == "right":
        for i, r in ctrls:
            if r == openvr.TrackedControllerRole_RightHand:
                return i, ctrls
    if prefer == "left":
        for i, r in ctrls:
            if r == openvr.TrackedControllerRole_LeftHand:
                return i, ctrls
    return ctrls[0][0], ctrls

def _bits_set(x: int):
    out = []
    b = 0
    while x:
        if x & 1:
            out.append(b)
        x >>= 1
        b += 1
    return out

def _axis_tuple(state, i):
    try:
        a = state.rAxis[i]
        return float(a.x), float(a.y)
    except Exception:
        return 0.0, 0.0


# -------------------- Open3D helpers --------------------

def make_grid(size=3.0, step=0.25):
    # grid in XZ plane (Y up), like many VR mental models
    lines = []
    points = []
    colors = []

    coords = np.arange(-size, size + 1e-9, step)

    def add_line(p0, p1, major=False):
        idx0 = len(points)
        points.append(p0)
        points.append(p1)
        lines.append([idx0, idx0 + 1])
        colors.append([0.6, 0.6, 0.6] if major else [0.35, 0.35, 0.35])

    for c in coords:
        major = abs(c) < 1e-9 or abs((c / step) % 4) < 1e-9
        # vertical lines along Z at X=c
        add_line([c, 0.0, -size], [c, 0.0, size], major=major)
        # horizontal lines along X at Z=c
        add_line([-size, 0.0, c], [size, 0.0, c], major=major)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls

def apply_pose(geom, T):
    # Set absolute pose by resetting then transforming
    # (Open3D doesn’t have set_transform; easiest stable method is rebuild)
    # For meshes, we can keep a pristine copy and overwrite geometry.
    geom.transform(T)


# -------------------- Live viewer --------------------

def main(
    prefer_hand="right",
    universe=openvr.TrackingUniverseSeated,
    hmd_relative=False,
    show_hmd=True,
    fps=90,
    pos_visual_scale=3.0,   # scale up translation so you can SEE motion easily
):
    openvr.init(openvr.VRApplication_Other)
    vrsys = openvr.VRSystem()

    # Try compositor (more reliable live poses on some Windows setups)
    compositor = None
    try:
        compositor = openvr.VRCompositor()
    except Exception:
        compositor = None

    cid, ctrls = _pick_controller(vrsys, prefer=prefer_hand)
    if cid is None:
        print("No controller found. Turn on SteamVR + controller.")
        openvr.shutdown()
        return

    print(f"Using controller id={cid} (prefer={prefer_hand}).")
    print(f"Pose source: {'VRCompositor.waitGetPoses' if compositor else 'VRSystem.getDeviceToAbsoluteTrackingPose'}")
    print("Viewer controls: left-drag orbit, right-drag pan, wheel zoom.\n")
    print("Button mapping help:")
    print("- Press ONE button at a time; this prints which pressed-bit changed.\n")

    # Open3D window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="OpenVR Controller Live Viewer", width=1280, height=800)

    grid = make_grid(size=3.0, step=0.25)
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    # Controller frame + a visible marker sphere at controller position
    ctrl_frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18)
    ctrl_sphere0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    ctrl_sphere0.compute_vertex_normals()

    hmd_frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18)
    hmd_sphere0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    hmd_sphere0.compute_vertex_normals()

    # We will *recreate* transformed meshes each frame from pristine copies (simple + reliable)
    ctrl_frame = ctrl_frame0
    ctrl_sphere = ctrl_sphere0
    hmd_frame = hmd_frame0
    hmd_sphere = hmd_sphere0

    vis.add_geometry(grid)
    vis.add_geometry(world_axes)
    vis.add_geometry(ctrl_frame)
    vis.add_geometry(ctrl_sphere)
    if show_hmd:
        vis.add_geometry(hmd_frame)
        vis.add_geometry(hmd_sphere)

    last_pressed = None
    last_axes = [(0.0, 0.0)] * 5
    last_pose_print = 0.0

    dt = 1.0 / max(1.0, float(fps))

    poses_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_type()
    game_poses = poses_type()

    try:
        while True:
            # --- get freshest poses ---
            if compositor is not None:
                try:
                    compositor.waitGetPoses(render_poses, game_poses)
                    poses = render_poses
                except Exception:
                    poses = vrsys.getDeviceToAbsoluteTrackingPose(
                        universe, 0, openvr.k_unMaxTrackedDeviceCount
                    )
            else:
                poses = vrsys.getDeviceToAbsoluteTrackingPose(
                    universe, 0, openvr.k_unMaxTrackedDeviceCount
                )

            pose_c = poses[cid]
            if not (pose_c.bDeviceIsConnected and pose_c.bPoseIsValid):
                vis.poll_events()
                vis.update_renderer()
                time.sleep(dt)
                continue

            Rw_c, pw_c = _mat34_to_Rp(pose_c.mDeviceToAbsoluteTracking)
            T_w_c = _T_from_Rp(Rw_c, pw_c)

            # Optional HMD-relative
            if hmd_relative:
                hid = openvr.k_unTrackedDeviceIndex_Hmd
                pose_h = poses[hid]
                if pose_h.bDeviceIsConnected and pose_h.bPoseIsValid:
                    Rw_h, pw_h = _mat34_to_Rp(pose_h.mDeviceToAbsoluteTracking)
                    T_w_h = _T_from_Rp(Rw_h, pw_h)
                    T_w_c = np.linalg.inv(T_w_h) @ T_w_c

            # Scale translation for visibility
            T_vis_c = T_w_c.copy()
            T_vis_c[:3, 3] *= float(pos_visual_scale)

            # --- rebuild controller meshes at new pose (stable approach) ---
            vis.remove_geometry(ctrl_frame, reset_bounding_box=False)
            vis.remove_geometry(ctrl_sphere, reset_bounding_box=False)

            ctrl_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18)
            ctrl_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            ctrl_sphere.compute_vertex_normals()

            ctrl_frame.transform(T_vis_c)
            ctrl_sphere.transform(T_vis_c)

            vis.add_geometry(ctrl_frame, reset_bounding_box=False)
            vis.add_geometry(ctrl_sphere, reset_bounding_box=False)

            # --- HMD pose optional ---
            if show_hmd:
                hid = openvr.k_unTrackedDeviceIndex_Hmd
                pose_h = poses[hid]
                if pose_h.bDeviceIsConnected and pose_h.bPoseIsValid:
                    Rw_h, pw_h = _mat34_to_Rp(pose_h.mDeviceToAbsoluteTracking)
                    T_w_h = _T_from_Rp(Rw_h, pw_h)
                    if hmd_relative:
                        T_w_h = np.eye(4)
                    T_vis_h = T_w_h.copy()
                    T_vis_h[:3, 3] *= float(pos_visual_scale)

                    vis.remove_geometry(hmd_frame, reset_bounding_box=False)
                    vis.remove_geometry(hmd_sphere, reset_bounding_box=False)

                    hmd_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18)
                    hmd_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    hmd_sphere.compute_vertex_normals()

                    hmd_frame.transform(T_vis_h)
                    hmd_sphere.transform(T_vis_h)

                    vis.add_geometry(hmd_frame, reset_bounding_box=False)
                    vis.add_geometry(hmd_sphere, reset_bounding_box=False)

            # --- buttons/axes (print only on change) ---
            _, state = vrsys.getControllerState(cid)
            pressed = int(state.ulButtonPressed)
            axes = [_axis_tuple(state, i) for i in range(5)]

            pressed_changed = (last_pressed is None) or (pressed != last_pressed)

            axis_move = False
            for i in range(5):
                dx = axes[i][0] - last_axes[i][0]
                dy = axes[i][1] - last_axes[i][1]
                if abs(dx) > 0.2 or abs(dy) > 0.2:
                    axis_move = True
                    break

            now = time.time()
            if pressed_changed:
                print(f"Pressed mask={pressed} bits={_bits_set(pressed)}")
            if axis_move and (now - last_pose_print > 0.1):
                last_pose_print = now
                print("Axes rAxis[0..4] =", [(round(a[0], 3), round(a[1], 3)) for a in axes])

            last_pressed = pressed
            last_axes = axes

            # --- render ---
            vis.poll_events()
            vis.update_renderer()

            # Debug: show the RAW position occasionally so we know it's changing
            if now - last_pose_print > 2.0:
                last_pose_print = now
                print(f"RAW controller pos (m) = {pw_c.round(4)}  (visual scale x{pos_visual_scale})")

            time.sleep(dt)

    finally:
        vis.destroy_window()
        openvr.shutdown()


if __name__ == "__main__":
    main(
        prefer_hand="right",
        universe=openvr.TrackingUniverseStanding,
        hmd_relative=False,
        show_hmd=True,
        fps=90,
        pos_visual_scale=3.0,
    )
