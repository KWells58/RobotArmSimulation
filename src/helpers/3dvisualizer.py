#!/usr/bin/env python3
"""
vive_viz_openvr.py

Live 3D visualization of a Vive Pro 2 HMD + controllers using OpenVR (SteamVR).
- Plots device positions in the OpenVR "standing" universe (meters).
- Shows orientation as RGB axis lines (X=red, Y=green, Z=blue).
- HMD + any tracked controllers (and optionally other trackers, if present).

Usage:
  python vive_viz_openvr.py

Requirements:
  pip install openvr numpy matplotlib
  SteamVR running + headset connected.
"""

import time
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ---------- OpenVR helpers ----------

def _safe_import_openvr():
    try:
        import openvr  # type: ignore
        return openvr
    except Exception as e:
        print("Failed to import openvr. Install with: pip install openvr")
        raise e


def _mat34_to_numpy(m) -> np.ndarray:
    """
    Robustly convert OpenVR HmdMatrix34_t (or similar) into a (3,4) float64 numpy array.
    Works across different openvr bindings where np.array(m) may produce a 0-d object array.
    """
    # Some bindings expose .m as the underlying 3x4 data
    if hasattr(m, "m"):
        try:
            arr = np.array(m.m, dtype=np.float64)
            if arr.shape == (3, 4):
                return arr
        except Exception:
            pass

    # Some bindings allow direct np.array conversion properly
    try:
        arr = np.array(m, dtype=np.float64)
        if arr.shape == (3, 4):
            return arr
        # Sometimes it's flattened length 12
        if arr.size == 12:
            return arr.reshape(3, 4)
    except Exception:
        pass

    # Final fallback: element-by-element indexing (works for ctypes structs)
    try:
        out = np.zeros((3, 4), dtype=np.float64)
        for r in range(3):
            for c in range(4):
                out[r, c] = float(m[r][c])
        return out
    except Exception:
        pass

    # Another fallback: m.m[r][c]
    try:
        out = np.zeros((3, 4), dtype=np.float64)
        mm = m.m  # type: ignore[attr-defined]
        for r in range(3):
            for c in range(4):
                out[r, c] = float(mm[r][c])
        return out
    except Exception as e:
        raise TypeError(f"Could not convert OpenVR matrix to numpy (3x4). Got type={type(m)}") from e


def _mat34_to_pose(m) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert OpenVR 3x4 device-to-absolute matrix into:
      - position (3,)
      - rotation matrix (3,3)
    """
    mat34 = _mat34_to_numpy(m)          # <-- key fix
    R = mat34[:, :3]
    t = mat34[:, 3]
    return t.astype(np.float64), R.astype(np.float64)


def _is_valid_pose(pose) -> bool:
    # OpenVR pose has fields: bPoseIsValid, bDeviceIsConnected, eTrackingResult, mDeviceToAbsoluteTracking
    return bool(getattr(pose, "bPoseIsValid", False)) and bool(getattr(pose, "bDeviceIsConnected", False))


def _device_class_name(openvr, device_class: int) -> str:
    if device_class == openvr.TrackedDeviceClass_HMD:
        return "HMD"
    if device_class == openvr.TrackedDeviceClass_Controller:
        return "Controller"
    if device_class == openvr.TrackedDeviceClass_GenericTracker:
        return "Tracker"
    if device_class == openvr.TrackedDeviceClass_TrackingReference:
        return "Base"
    return f"Unknown({device_class})"


def _role_name(openvr, role: int) -> str:
    # Roles can be invalid (0), left (1), right (2) depending on bindings.
    if hasattr(openvr, "TrackedControllerRole_LeftHand") and role == openvr.TrackedControllerRole_LeftHand:
        return "Left"
    if hasattr(openvr, "TrackedControllerRole_RightHand") and role == openvr.TrackedControllerRole_RightHand:
        return "Right"
    return "UnknownRole"


def _get_string_property(vr, openvr, device_index: int, prop_enum) -> str:
    """
    Read a device string property (like model/serial).
    """
    try:
        err = openvr.TrackedPropertyError.TrackedProp_Success
        s = vr.getStringTrackedDeviceProperty(device_index, prop_enum)
        return str(s)
    except Exception:
        return ""


# ---------- Visualization helpers ----------

@dataclass
class DeviceViz:
    label: str
    color: str
    point_artist: any
    text_artist: any
    axis_artists: List[any]  # 3 line objects (x,y,z)


def _set_3d_limits_equal(ax, center: np.ndarray, radius: float):
    cx, cy, cz = center.tolist()
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(cz - radius, cz + radius)


def _draw_grid_floor(ax, size=2.0, step=0.5, z=0.0, alpha=0.2):
    # simple XY grid at z=0
    xs = np.arange(-size, size + 1e-6, step)
    ys = np.arange(-size, size + 1e-6, step)
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], [z, z], linewidth=0.5, alpha=alpha)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], [z, z], linewidth=0.5, alpha=alpha)


def _update_point(artist, p: np.ndarray):
    artist.set_data([p[0]], [p[1]])
    artist.set_3d_properties([p[2]])


def _update_text(text_artist, p: np.ndarray, s: str):
    text_artist.set_position((p[0], p[1]))
    text_artist.set_3d_properties(p[2])
    text_artist.set_text(s)


def _update_axes(axis_artists, p: np.ndarray, R: np.ndarray, axis_len: float):
    """
    axis_artists: [x_line, y_line, z_line]
    draws 3 axis lines from position p using columns of R as basis vectors
    """
    # OpenVR uses right-handed coordinates; in standing universe this is usually:
    # +X right, +Y up, -Z forward (common in OpenVR). We'll just draw what we get.
    # Columns of R correspond to basis vectors in world coords.
    ex = R[:, 0]
    ey = R[:, 1]
    ez = R[:, 2]

    ends = [p + axis_len * ex, p + axis_len * ey, p + axis_len * ez]

    for line, end in zip(axis_artists, ends):
        line.set_data([p[0], end[0]], [p[1], end[1]])
        line.set_3d_properties([p[2], end[2]])


# ---------- Main loop ----------

def main():
    openvr = _safe_import_openvr()

    # Initialize OpenVR
    try:
        openvr.init(openvr.VRApplication_Scene)
    except Exception as e:
        print("OpenVR init failed. Is SteamVR running and headset connected?")
        raise e

    vr = openvr.VRSystem()

    # Ask for "standing" tracking space
    tracking_space = openvr.TrackingUniverseStanding

    # Matplotlib 3D setup
    plt.ion()
    fig = plt.figure("Vive (OpenVR) 3D Visualizer", figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("OpenVR Tracked Devices (HMD + Controllers)")

    _draw_grid_floor(ax, size=3.0, step=0.5, z=0.0, alpha=0.2)

    # Legend colors
    HMD_COLOR = "black"
    LEFT_COLOR = "blue"
    RIGHT_COLOR = "orange"
    OTHER_CTRL_COLOR = "purple"

    # Cache visualization objects per device index
    viz: Dict[int, DeviceViz] = {}

    # Size of orientation axes
    axis_len = 0.12

    # Plot radius around HMD
    view_radius = 2.0

    # Simple loop
    try:
        while plt.fignum_exists(fig.number):
            poses = vr.getDeviceToAbsoluteTrackingPose(tracking_space, 0, openvr.k_unMaxTrackedDeviceCount)

            # Try to keep view centered on HMD if available
            hmd_pos = None

            for i in range(openvr.k_unMaxTrackedDeviceCount):
                pose = poses[i]
                if not _is_valid_pose(pose):
                    continue

                dev_class = vr.getTrackedDeviceClass(i)
                dev_class_str = _device_class_name(openvr, dev_class)

                # Get transform
                pos, R = _mat34_to_pose(pose.mDeviceToAbsoluteTracking)

                # Identify device labeling
                label = dev_class_str
                color = "gray"

                if dev_class == openvr.TrackedDeviceClass_HMD:
                    label = "HMD"
                    color = HMD_COLOR
                    hmd_pos = pos

                elif dev_class == openvr.TrackedDeviceClass_Controller:
                    # Determine controller role if possible
                    try:
                        role = vr.getControllerRoleForTrackedDeviceIndex(i)
                    except Exception:
                        role = 0

                    rname = _role_name(openvr, role)
                    if rname == "Left":
                        label = "Controller (Left)"
                        color = LEFT_COLOR
                    elif rname == "Right":
                        label = "Controller (Right)"
                        color = RIGHT_COLOR
                    else:
                        label = "Controller"
                        color = OTHER_CTRL_COLOR

                else:
                    # Skip bases by default (comment this out if you want them drawn)
                    if dev_class == openvr.TrackedDeviceClass_TrackingReference:
                        continue
                    label = dev_class_str
                    color = "gray"

                # Make a slightly richer label with model/serial if you want
                # model = _get_string_property(vr, openvr, i, openvr.Prop_ModelNumber_String)
                # serial = _get_string_property(vr, openvr, i, openvr.Prop_SerialNumber_String)
                # if model or serial:
                #     label = f"{label}\n{model} {serial}".strip()

                # Create artists if needed
                if i not in viz:
                    point = ax.plot([pos[0]], [pos[1]], [pos[2]], marker="o", markersize=8, linestyle="")[0]
                    point.set_color(color)

                    text = ax.text(pos[0], pos[1], pos[2], label, fontsize=9)

                    # axis lines (X,Y,Z) - matplotlib uses per-line color
                    x_line = ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2]], linewidth=2)[0]
                    y_line = ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2]], linewidth=2)[0]
                    z_line = ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2], pos[2]], linewidth=2)[0]
                    x_line.set_color("red")
                    y_line.set_color("green")
                    z_line.set_color("blue")

                    viz[i] = DeviceViz(label=label, color=color, point_artist=point, text_artist=text,
                                       axis_artists=[x_line, y_line, z_line])

                # Update artists
                dv = viz[i]
                dv.point_artist.set_color(color)
                _update_point(dv.point_artist, pos)
                _update_text(dv.text_artist, pos, label)
                _update_axes(dv.axis_artists, pos, R, axis_len)

            # Hide devices that disappeared (optional lightweight cleanup)
            # If you want to actually remove, you'd need ax.lines removal management;
            # for now we just don't update them, leaving last pose visible.

            # Center view around HMD
            if hmd_pos is not None:
                _set_3d_limits_equal(ax, center=hmd_pos, radius=view_radius)

            plt.pause(0.01)  # ~100 Hz UI responsiveness (actual pose rate depends on SteamVR)
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            openvr.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()