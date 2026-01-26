# src/main.py
import time
import openvr

from src.controllers import ViveController
from src.simulation import create_environment, apply_action


def pick_vive_controller(prefer_role: str = "right") -> int:
    """
    Returns a tracked device index for a Vive controller.
    Prefers right/left hand role if SteamVR reports it, otherwise falls back to first controller found.
    """
    vr = openvr.VRSystem()

    ROLE_LEFT = getattr(openvr, "TrackedControllerRole_LeftHand", 1)
    ROLE_RIGHT = getattr(openvr, "TrackedControllerRole_RightHand", 2)
    target_role = ROLE_RIGHT if prefer_role.lower() == "right" else ROLE_LEFT

    # 1) role-based match (best)
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            role = vr.getControllerRoleForTrackedDeviceIndex(i)
            if role == target_role:
                return i

    # 2) fallback: first controller found
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
            return i

    raise RuntimeError("No Vive controller found (TrackedDeviceClass_Controller). Is SteamVR running?")


def main():
    print("[DEBUG] starting...", flush=True)

    openvr.init(openvr.VRApplication_Scene)
    print("[DEBUG] openvr.init OK", flush=True)

    poses_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_type()
    game_poses = poses_type()

    # Must call once so VRSystem/roles are valid in some setups
    openvr.VRCompositor().waitGetPoses(render_poses, game_poses)

    controller_id = pick_vive_controller(prefer_role="right")
    print(f"[DEBUG] Using Vive controller device_id={controller_id}", flush=True)

    env, obs = create_environment()
    env.render()

    vr = ViveController(device_id=controller_id)

    print(
        "\n[VR] Teleoperation started.\n"
        "Trackpad XY -> planar\n"
        "Hold GRIP -> enable Z (controller up/down)\n"
        "Trigger -> toggle gripper\n"
        "Menu -> reset origin\n",
        flush=True,
    )

    try:
        while True:
            openvr.VRCompositor().waitGetPoses(render_poses, game_poses)
            state = vr.update(render_poses)  # <-- important: pass poses

            if state is not None:
                obs, _, _, _ = apply_action(env, state)

            env.render()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
