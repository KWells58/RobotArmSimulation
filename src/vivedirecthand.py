import numpy as np
import openvr
import time
import robosuite
from robosuite.controllers import load_part_controller_config as load_controller_config
# ============================================================
# Simplified Vive Controller Wrapper (no wrist rotation)
# ============================================================

class ViveController:
    """
    Simplified for planar movement only:
        - Trackpad XY -> X/Y
        - Controller Z -> Z
        - Trigger -> gripper toggle
        - Menu -> reset origin
    """
    def __init__(self, device_id=4, pos_scale=0.80, z_scale=5, smooth=0.2):
        self.device_id = device_id
        self.pos_scale = pos_scale  # planar scale
        self.z_scale = z_scale      # vertical scale
        self.smooth = smooth

        self.s_dpos = np.zeros(3)
        self.origin_pos = None

        self.last_trigger = False
        self.gripper = -1
        self.reset = False

    def _smooth_val(self, prev, new):
        return self.smooth * new + (1 - self.smooth) * prev

    def update(self, poses):
        pose = poses[self.device_id]
        if not pose.bPoseIsValid:
            return None

        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3]

        # Initialize origin
        if self.origin_pos is None:
            self.origin_pos = pos.copy()
            return None

        # Controller state
        state = openvr.VRSystem().getControllerState(self.device_id)[1]

        # Trackpad XY (non-linear for better sensitivity)
        track_x = state.rAxis[0].x
        track_y = state.rAxis[0].y
        planar = np.array([
            np.sign(track_x) * (abs(track_x)**0.8) * self.pos_scale,
            np.sign(track_y) * (abs(track_y)**0.8) * self.pos_scale,
            0.0
        ])

        # Vertical Z
        z = (pos[2] - self.origin_pos[2]) * self.z_scale
        dpos = np.array([planar[1], planar[0], z]) #DONOT CHANGE 1 AND 0
        self.s_dpos = self._smooth_val(self.s_dpos, dpos)

        # Trigger -> toggle gripper
        trigger_down = state.rAxis[1].x > 0.8
        if trigger_down and not self.last_trigger:
            self.gripper = -self.gripper
        self.last_trigger = trigger_down

        # Menu button -> reset origin
        k_EButton_ApplicationMenu = 2
        self.reset = (state.ulButtonPressed & (1 << k_EButton_ApplicationMenu)) != 0
        if self.reset:
            self.origin_pos = pos.copy()

        return dict(
            dpos=self.s_dpos.copy(),
            grasp=self.gripper,
            reset=self.reset
        )



# ============================================================
# Main Teleoperation Loop
# ============================================================

def main():
    openvr.init(openvr.VRApplication_Scene)
    poses_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    render_poses = poses_type()
    game_poses = poses_type()

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(
        "Lift",
        robots=["UR5e"],
        controller_configs=controller_config,
        gripper_types="default",
        env_configuration="single-arm-opposed",
        has_renderer=True,
        render_camera="frontview",
        control_freq=20,
        ignore_done=True,
        horizon=10000,
        use_object_obs=False,
        use_camera_obs=False,
    )

    obs = env.reset()
    env.render()

    vr = ViveController(device_id=4)
    print("\n[VR] Teleoperation started. Trackpad XY -> planar, Z -> vertical.\n")

    try:
        while True:
            openvr.VRCompositor().waitGetPoses(render_poses, game_poses)
            state = vr.update(render_poses)

            if state is not None:
                # Action: [dpos_x, dpos_y, dpos_z, grasp]
                #drot zero for now to isolate planar movement
                drot = np.zeros(3)
                action = np.concatenate([state["dpos"], drot, [state["grasp"]]])
                obs, _, _, _ = env.step(action)

            env.render()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
