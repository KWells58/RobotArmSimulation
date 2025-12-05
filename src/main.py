import time
import openvr
from src.controllers.vive_controller import ViveController
from src.simulation import create_environment, apply_action


def main():
    # Initialize OpenVR
    openvr.init(openvr.VRApplication_Scene)

    try:
        # Create Robosuite environment
        env, obs = create_environment()
        env.render()

        # Initialize ViveController (Robosuite-style)
        vr = ViveController(env=env, device_id=4)
        vr.start_control()

        print("\n[VR] Teleoperation started. Trackpad XY -> planar, Z -> vertical.\n")

        # Main control loop
        while True:
            state = vr.get_controller_state()  # gets smoothed 6-DOF + grasp + reset

            if state is not None:
                # Apply action to the environment
                obs, reward, done, info = apply_action(env, state)

            env.render()
            time.sleep(0.01)  # 100 Hz control loop

    except KeyboardInterrupt:
        print("\nExiting VR teleoperation...")

    finally:
        openvr.shutdown()
        env.close()


if __name__ == "__main__":
    main()
