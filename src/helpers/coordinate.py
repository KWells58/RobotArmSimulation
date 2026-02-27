# coordinate.py
# Windows-friendly "mini RViz" viewer for Vive / SteamVR controllers using OpenVR + Open3D (0.19+)
#
# What you get:
# - A simple "room": floor grid + two walls
# - World axes arrows at the origin (+X red, +Y green, +Z blue)
# - Controller coordinate frames for up to two controllers
# - A bright "forward ray" attached to each controller to make orientation obvious
# - Robust controller discovery (doesn't rely on left/right roles)
#
# Install:
#   pip install openvr open3d numpy
#
# Run:
#   python coordinate.py
#
# Notes:
# - If nothing moves, make sure SteamVR is running and controllers are on.
# - If the ray points "backwards" relative to your controller, flip FORWARD_AXIS below.

import time
import numpy as np
import openvr
import open3d as o3d


# ---------------------------
# Tuning
# ---------------------------

RATE_HZ = 90.0

# Controller "forward" axis in the controller's local frame.
# Try "z" first. If it doesn't match your physical intuition, change to "x" or "-z", "-x".
FORWARD_AXIS = "z"   # one of: "x", "y", "z", "-x", "-y", "-z"
FORWARD_LEN = 0.35

# Room visuals
ROOM_SIZE = 2.5     # half-extent (meters)
ROOM_HEIGHT = 2.0
GRID_STEP = 0.25

# Optional basis remap for preview (leave identity first)
# Example mapping you used earlier:
#   x_mj = +x_vr
#   y_mj = -z_vr
#   z_mj = +y_vr
A_PREVIEW = np.eye(3, dtype=np.float64)
# A_PREVIEW = np.array([[1, 0, 0],
#                       [0, 0, -1],
#                       [0, 1, 0]], dtype=np.float64)


# ---------------------------
# OpenVR helpers
# ---------------------------

def openvr_mat34_to_Rp(mat34):
    """mat34 is openvr.HmdMatrix34_t (12 floats row-major): returns (R(3,3), p(3,))."""
    M = np.array(list(mat34), dtype=np.float64).reshape(3, 4)
    R = M[:, :3].copy()
    p = M[:, 3].copy()
    return R, p


def apply_preview_basis(R, p):
    """Change basis with A: p' = A p, R' = A R A^T."""
    A = A_PREVIEW
    p2 = A @ p
    R2 = A @ R @ A.T
    return R2, p2


def list_all_controllers(vrsys):
    """Return a list of tracked device indices that are controllers."""
    ctrls = []
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vrsys.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            ctrls.append(i)
    return ctrls


def pick_two_controllers(vrsys):
    """
    Robust pick:
      - Prefer left/right roles if available
      - Otherwise choose first two controllers found
    Returns (id0, id1, all_ctrls)
    """
    all_ctrls = list_all_controllers(vrsys)
    left = None
    right = None

    for i in all_ctrls:
        role = vrsys.getControllerRoleForTrackedDeviceIndex(i)
        if role == openvr.TrackedControllerRole_LeftHand:
            left = i
        elif role == openvr.TrackedControllerRole_RightHand:
            right = i

    chosen = []
    if left is not None:
        chosen.append(left)
    if right is not None and right != left:
        chosen.append(right)

    # Fill remainder with any other controllers
    for i in all_ctrls:
        if len(chosen) >= 2:
            break
        if i not in chosen:
            chosen.append(i)

    id0 = chosen[0] if len(chosen) >= 1 else None
    id1 = chosen[1] if len(chosen) >= 2 else None
    return id0, id1, all_ctrls


def device_serial(vrsys, idx):
    """Best-effort device serial string."""
    try:
        return vrsys.getStringTrackedDeviceProperty(idx, openvr.Prop_SerialNumber_String)
    except Exception:
        return "?"


# ---------------------------
# Open3D geometry helpers
# ---------------------------

def make_room(size=2.5, height=2.0, step=0.25):
    """
    Simple room visualization:
      - floor grid (z=0)
      - back wall (y=+size)
      - side wall (x=+size)
    """
    geoms = []

    # Floor grid
    pts = []
    lines = []
    colors = []
    n = int(size / step)

    def add_line(p0, p1, color):
        pts.append(p0); pts.append(p1)
        lines.append([len(pts)-2, len(pts)-1])
        colors.append(color)

    for i in range(-n, n + 1):
        x = i * step
        add_line([x, -size, 0], [x, +size, 0], [0.60, 0.60, 0.60])
        y = i * step
        add_line([-size, y, 0], [+size, y, 0], [0.60, 0.60, 0.60])

    floor = o3d.geometry.LineSet()
    floor.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    floor.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    floor.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    geoms.append(floor)

    # Wall grids
    def make_wall(axis="y", value=+size):
        wpts, wlines, wcolors = [], [], []
        # vertical lines (in wall plane)
        for i in range(-n, n + 1):
            t = i * step
            if axis == "y":
                wpts.extend([[t, value, 0], [t, value, height]])
            else:  # axis == "x"
                wpts.extend([[value, t, 0], [value, t, height]])
            wlines.append([len(wpts)-2, len(wpts)-1])
            wcolors.append([0.35, 0.35, 0.35])

        # horizontal lines (vary height)
        m = 10
        for j in range(m + 1):
            h = (j / m) * height
            if axis == "y":
                wpts.extend([[-size, value, h], [+size, value, h]])
            else:
                wpts.extend([[value, -size, h], [value, +size, h]])
            wlines.append([len(wpts)-2, len(wpts)-1])
            wcolors.append([0.35, 0.35, 0.35])

        wall = o3d.geometry.LineSet()
        wall.points = o3d.utility.Vector3dVector(np.array(wpts, dtype=np.float64))
        wall.lines = o3d.utility.Vector2iVector(np.array(wlines, dtype=np.int32))
        wall.colors = o3d.utility.Vector3dVector(np.array(wcolors, dtype=np.float64))
        return wall

    geoms.append(make_wall(axis="y", value=+size))  # back wall
    geoms.append(make_wall(axis="x", value=+size))  # side wall
    return geoms


def make_world_axes(scale=0.6):
    """Line arrows for +X, +Y, +Z at origin."""
    pts = np.array([
        [0, 0, 0], [scale, 0, 0],   # +X
        [0, 0, 0], [0, scale, 0],   # +Y
        [0, 0, 0], [0, 0, scale],   # +Z
    ], dtype=np.float64)
    lines = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    colors = np.array([
        [1.0, 0.2, 0.2],  # X (red-ish)
        [0.2, 1.0, 0.2],  # Y (green-ish)
        [0.2, 0.4, 1.0],  # Z (blue-ish)
    ], dtype=np.float64)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_forward_ray(length=0.35):
    """
    A bright ray along a chosen local axis.
    We'll attach this to controller frame via transforms.
    """
    axis = FORWARD_AXIS.lower().strip()
    v = np.zeros(3, dtype=np.float64)
    sign = 1.0
    if axis.startswith("-"):
        sign = -1.0
        axis = axis[1:]
    if axis == "x":
        v[0] = sign * length
    elif axis == "y":
        v[1] = sign * length
    else:
        v[2] = sign * length

    pts = np.array([[0, 0, 0], v], dtype=np.float64)
    lines = np.array([[0, 1]], dtype=np.int32)

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 0.2]], dtype=np.float64))
    return ls


def o3d_transform_from_Rp(R, p):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# ---------------------------
# Main
# ---------------------------

def main():
    openvr.init(openvr.VRApplication_Other)
    vrsys = openvr.VRSystem()

    vis = o3d.visualization.Visualizer() # type: ignore
    vis.create_window(window_name="Vive Room Viewer (Open3D)", width=1400, height=900)

    # Room + world axes
    for g in make_room(size=ROOM_SIZE, height=ROOM_HEIGHT, step=GRID_STEP):
        vis.add_geometry(g)
    world_axes = make_world_axes(scale=0.7)
    vis.add_geometry(world_axes)

    # A coordinate frame at origin (classic)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    vis.add_geometry(origin_frame)

    # Controller frames + forward rays
    ctrl0_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18, origin=[0, 0, 0])
    ctrl1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18, origin=[0, 0, 0])
    ctrl0_ray = make_forward_ray(length=FORWARD_LEN)
    ctrl1_ray = make_forward_ray(length=FORWARD_LEN)

    vis.add_geometry(ctrl0_frame)
    vis.add_geometry(ctrl1_frame)
    vis.add_geometry(ctrl0_ray)
    vis.add_geometry(ctrl1_ray)

    # Camera: make it feel like a room
    ctr = vis.get_view_control()
    ctr.set_front([0.35, -1.0, 0.55])
    ctr.set_lookat([0.0, 0.0, 0.8])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.70)

    dt = 1.0 / max(1.0, float(RATE_HZ))

    # Cache last transforms in Python variables (Open3D meshes can't store attrs reliably)
    last_T0 = np.eye(4, dtype=np.float64)
    last_T1 = np.eye(4, dtype=np.float64)
    last_T0_ray = np.eye(4, dtype=np.float64)
    last_T1_ray = np.eye(4, dtype=np.float64)
    init0 = init1 = False
    init0_ray = init1_ray = False

    last_dbg = 0.0

    print("Running. Tips:")
    print("- Make sure SteamVR is running and controllers are ON.")
    print("- Orbit with left mouse, pan with right mouse, zoom with wheel.")
    print(f"- Controller forward ray axis = {FORWARD_AXIS} (edit FORWARD_AXIS at top if wrong).")
    print("- If axes feel flipped vs MuJoCo, set A_PREVIEW near top.\n")

    try:
        while True:
            poses = vrsys.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0,
                openvr.k_unMaxTrackedDeviceCount
            )

            id0, id1, all_ctrls = pick_two_controllers(vrsys)

            # Debug print once per second
            now = time.time()
            if now - last_dbg > 1.0:
                last_dbg = now
                print("\n--- DEBUG ---")
                print("controllers:", all_ctrls)
                for i in all_ctrls[:4]:
                    p = poses[i]
                    print(
                        f"  id={i:2d} connected={int(p.bDeviceIsConnected)} valid={int(p.bPoseIsValid)} "
                        f"serial={device_serial(vrsys, i)}"
                    )

            # Update controller 0
            if id0 is not None and poses[id0].bPoseIsValid:
                R, p = openvr_mat34_to_Rp(poses[id0].mDeviceToAbsoluteTracking)
                R, p = apply_preview_basis(R, p)
                T = o3d_transform_from_Rp(R, p)

                if not init0:
                    ctrl0_frame.transform(T)
                    last_T0 = T
                    init0 = True
                else:
                    delta = T @ np.linalg.inv(last_T0)
                    ctrl0_frame.transform(delta)
                    last_T0 = T

                if not init0_ray:
                    ctrl0_ray.transform(T)
                    last_T0_ray = T
                    init0_ray = True
                else:
                    delta_ray = T @ np.linalg.inv(last_T0_ray)
                    ctrl0_ray.transform(delta_ray)
                    last_T0_ray = T

            # Update controller 1
            if id1 is not None and poses[id1].bPoseIsValid:
                R, p = openvr_mat34_to_Rp(poses[id1].mDeviceToAbsoluteTracking)
                R, p = apply_preview_basis(R, p)
                T = o3d_transform_from_Rp(R, p)

                if not init1:
                    ctrl1_frame.transform(T)
                    last_T1 = T
                    init1 = True
                else:
                    delta = T @ np.linalg.inv(last_T1)
                    ctrl1_frame.transform(delta)
                    last_T1 = T

                if not init1_ray:
                    ctrl1_ray.transform(T)
                    last_T1_ray = T
                    init1_ray = True
                else:
                    delta_ray = T @ np.linalg.inv(last_T1_ray)
                    ctrl1_ray.transform(delta_ray)
                    last_T1_ray = T

            vis.update_geometry(ctrl0_frame)
            vis.update_geometry(ctrl1_frame)
            vis.update_geometry(ctrl0_ray)
            vis.update_geometry(ctrl1_ray)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(dt)

    finally:
        vis.destroy_window()
        openvr.shutdown()


if __name__ == "__main__":
    main()
