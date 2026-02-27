import time
import numpy as np
import openvr

from src.controllers import ViveController
from src.simulation import create_environment, apply_action


def pick_vive_controller(prefer_role: str = "right") -> int | None:
    """
    Returns a tracked device index for a Vive controller.
    Prefers right/left hand role if SteamVR reports it, otherwise falls back.
    """
    vr = openvr.VRSystem()

    role_left = getattr(openvr, "TrackedControllerRole_LeftHand", 1)
    role_right = getattr(openvr, "TrackedControllerRole_RightHand", 2)
    target_role = role_right if prefer_role.lower() == "right" else role_left

    # Prefer matching role
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            role = vr.getControllerRoleForTrackedDeviceIndex(i)
            if role == target_role:
                return i

    # Fallback: first controller found
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            return i

    return None


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array(
        [
            [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)],
        ],
        dtype=np.float64,
    )


def _get_ee_pose_from_obs(obs: dict, prefix: str):
    """
    prefix examples: "robot0", "robot1"
    Returns (ee_pos(3,), ee_rotmat(3,3))

    Works with typical robosuite key names:
      {prefix}_eef_pos
      {prefix}_eef_quat
      {prefix}_eef_rot   (if exists)
    """
    pos_key_candidates = [
        f"{prefix}_eef_pos",
        f"{prefix}_gripper_pos",
        "robot0_eef_pos",  # extra fallback
    ]
    rot_key_candidates = [
        f"{prefix}_eef_rot",
        f"{prefix}_gripper_rot",
    ]
    quat_key_candidates = [
        f"{prefix}_eef_quat",
        f"{prefix}_gripper_quat",
    ]

    ee_pos = None
    for k in pos_key_candidates:
        if k in obs:
            ee_pos = np.asarray(obs[k], dtype=np.float64).reshape(3)
            break

    ee_R = None
    for k in rot_key_candidates:
        if k in obs:
            ee_R = np.asarray(obs[k], dtype=np.float64).reshape(3, 3)
            break

    if ee_R is None:
        quat = None
        for k in quat_key_candidates:
            if k in obs:
                quat = np.asarray(obs[k], dtype=np.float64).reshape(4)
                break
        if quat is not None:
            ee_R = quat_xyzw_to_rotmat(quat)

    if ee_pos is None or ee_R is None:
        raise KeyError(f"Couldn't find EE pose for {prefix}. Available keys include: {list(obs.keys())[:30]} ...")

    return ee_pos, ee_R


def main():
    # -------- choose env here --------
    # Single arm:
    env_name = "Lift"
    robots = ("UR5e",)

    # Example dual-arm (ONLY if your robosuite has it):
    # env_name = "TwoArmLift"
    # robots = ("Panda", "Panda")

    print("[DEBUG] creating robosuite env...", flush=True)
    env, obs = create_environment(env_name=env_name, robots=robots)

    #home snapshot
    home_qpos = env.sim.data.qpos.copy()
    home_qvel = env.sim.data.qvel.copy()
    def restore_home():
        """Restore sim to start state, forward, and fetch fresh observations."""
        env.sim.data.qpos[:] = home_qpos
        env.sim.data.qvel[:] = home_qvel

        # Forward the physics so derived quantities update
        if hasattr(env.sim, "forward"):
            env.sim.forward()
        # Get a fresh obs without calling reset() (keeps episode running)
        if hasattr(env, "_get_observations"):
            return env._get_observations()
        # Fallback (rare)
        return env.reset()
    

    #print("action_dim:", env.action_dim)
    #if hasattr(env, "action_spec"):
    #    print("action_spec:", env.action_spec)
    #env.render()

    adim = int(getattr(env, "action_dim", 0))
    dual_arm = adim >= 14
    print(f"[DEBUG] env_name={env_name} robots={robots} action_dim={adim} dual_arm={dual_arm}", flush=True)

    # Init OpenVR once here (so role picking works reliably)
    openvr.init(openvr.VRApplication_Other)

    right_id = pick_vive_controller("right")
    left_id = pick_vive_controller("left")
    print(f"[DEBUG] Vive device ids: right={right_id} left={left_id}", flush=True)

    # Create controllers (hmd_relative helps desk setups)
    vr_right = ViveController(device_id=right_id, hmd_relative=True)
    vr_left = ViveController(device_id=left_id, hmd_relative=True) if dual_arm else None

    # Calibrate at start
    if not dual_arm:
        ee_pos, ee_R = _get_ee_pose_from_obs(obs, "robot0")
        vr_right.start_control(ee_pos, ee_R)
    else:
        eeR_pos, eeR_R = _get_ee_pose_from_obs(obs, "robot0")
        eeL_pos, eeL_R = _get_ee_pose_from_obs(obs, "robot1")
        vr_right.start_control(eeR_pos, eeR_R)
        if vr_left is not None:
            vr_left.start_control(eeL_pos, eeL_R)

    print(
        "\n[VR] Teleoperation started.\n"
        "Hold GRIP -> clutch enable\n"
        "Trigger -> toggle gripper\n"
        "Menu -> recalibrate (bind controller to current EE)\n",
        flush=True,
    )
    

    try:
        while True:
            if not dual_arm:
                ee_pos, ee_R = _get_ee_pose_from_obs(obs, "robot0")
                stR = vr_right.update(ee_pos, ee_R)

                # -------- HOME REQUEST (single arm) --------
                if stR is not None and stR.get("home", False):
                    obs = restore_home()
                    ee_pos, ee_R = _get_ee_pose_from_obs(obs, "robot0")
                    vr_right.start_control(ee_pos, ee_R)
                    continue
                # ------------------------------------------

                if stR is not None:
                    obs, _, _, _ = apply_action(env, stR)

            else:
                eeR_pos, eeR_R = _get_ee_pose_from_obs(obs, "robot0")
                eeL_pos, eeL_R = _get_ee_pose_from_obs(obs, "robot1")

                stR = vr_right.update(eeR_pos, eeR_R)
                stL = vr_left.update(eeL_pos, eeL_R) if vr_left is not None else None

                # -------- HOME REQUEST (dual arm) --------
                home_requested = (
                    (stR is not None and stR.get("home", False)) or
                    (stL is not None and stL.get("home", False))
                )
                if home_requested:
                    obs = restore_home()
                    eeR_pos, eeR_R = _get_ee_pose_from_obs(obs, "robot0")
                    eeL_pos, eeL_R = _get_ee_pose_from_obs(obs, "robot1")
                    vr_right.start_control(eeR_pos, eeR_R)
                    if vr_left is not None:
                        vr_left.start_control(eeL_pos, eeL_R)
                    continue
                # ----------------------------------------

                if stR is not None or stL is not None:
                    obs, _, _, _ = apply_action(env, stR, stL)

            env.render()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()