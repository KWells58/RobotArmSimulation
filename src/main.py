import time
import numpy as np

from src.controllers import ViveController
from src.simulation import create_environment, apply_action


def _get_ee_pose_from_obs(obs: dict):
    """
    Try common robosuite observation keys. Adjust if your obs uses different names.
    Returns (ee_pos(3,), ee_rotmat(3,3)).
    """
    # Common robosuite key patterns (varies by env/controller config)
    candidates_pos = [
        "robot0_eef_pos",
        "eef_pos",
        "robot0_gripper_pos",
    ]
    candidates_quat = [
        "robot0_eef_quat",
        "eef_quat",
        "robot0_gripper_quat",
    ]
    candidates_mat = [
        "robot0_eef_rot",
        "eef_rot",
        "robot0_gripper_rot",
    ]

    ee_pos = None
    for k in candidates_pos:
        if k in obs:
            ee_pos = np.asarray(obs[k], dtype=np.float64).reshape(3)
            break

    ee_rotmat = None
    # If rot matrix exists, use it
    for k in candidates_mat:
        if k in obs:
            ee_rotmat = np.asarray(obs[k], dtype=np.float64).reshape(3, 3)
            break

    # Else try quaternion -> matrix (xyzw is typical in robosuite; sometimes wxyz)
    if ee_rotmat is None:
        quat = None
        for k in candidates_quat:
            if k in obs:
                quat = np.asarray(obs[k], dtype=np.float64).reshape(4)
                break
        if quat is not None:
            # Heuristic: if |w| is last element and looks like a unit quat, assume xyzw.
            # If your env uses wxyz, swap accordingly.
            x, y, z, w = quat
            # quat -> rotmat
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            ee_rotmat = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
                [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
            ], dtype=np.float64)

    if ee_pos is None or ee_rotmat is None:
        raise KeyError(
            "Couldn't find EE pose in obs. Print(obs.keys()) and update _get_ee_pose_from_obs()."
        )

    return ee_pos, ee_rotmat


def main():
    print("[DEBUG] creating robosuite env...", flush=True)
    env, obs = create_environment()
    env.render()

    ee_pos, ee_rotmat = _get_ee_pose_from_obs(obs)

    # Vive controller polls OpenVR internally (no compositor)
    vr = ViveController(device_id=None, hmd_relative=True)

    # Calibrate once at start so controller pose is anchored to current EE pose
    vr.start_control(ee_pos, ee_rotmat)

    print(
        "\n[VR] Teleoperation started.\n"
        "Hold GRIP -> clutch enable (absolute pose anchored)\n"
        "Controller absolute pose -> target EE pose\n"
        "Trigger -> toggle gripper\n"
        "Menu -> recalibrate (bind controller to current EE)\n",
        flush=True,
    )

    try:
        while True:
            # Get latest EE pose
            ee_pos, ee_rotmat = _get_ee_pose_from_obs(obs)

            # Get VR command deltas for OSC_POSE
            state = vr.update(ee_pos, ee_rotmat)

            if state is not None:
                obs, _, _, _ = apply_action(env, state)

            env.render()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        # ViveController opens OpenVR internally; we shut it down there by letting process end.
        # If you want explicit cleanup, add vr.shutdown() method and call it here.
        pass


if __name__ == "__main__":
    main()
