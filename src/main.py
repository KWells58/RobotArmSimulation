import openvr
import time
from src.controllers import ViveController
from src.simulation import create_environment, apply_action

def main():
    openvr.init(openvr.VRApplication_Scene)

    poses_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_type()
    game_poses = poses_type()

    env, obs = create_environment()
    env.render()

    vr = ViveController(device_id=4)
    print("\n[VR] Teleoperation started. Trackpad XY -> planar, Z -> vertical.\n")

    try:
        while True:
            openvr.VRCompositor().waitGetPoses(render_poses, game_poses)
            state = vr.update(render_poses)

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
