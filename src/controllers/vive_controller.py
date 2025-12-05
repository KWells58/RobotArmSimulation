import numpy as np
import openvr
from robosuite.devices.device import Device

class ViveController(Device):
    def __init__(self, env, device_id=4, pos_scale=0.8, z_scale=5, smooth=0.2):
        super().__init__(env)
        self.device_id = device_id
        self.pos_scale = pos_scale
        self.z_scale = z_scale
        self.smooth = smooth

        self.s_dpos = np.zeros(3)
        self.origin_pos = None

        self.gripper = -1
        self.last_trigger = False
        self.reset = False

    def _smooth_val(self, prev, new):
        return self.smooth * new + (1 - self.smooth) * prev

    def start_control(self):
        """Required by Robosuite Device API."""
        pass

    def get_controller_state(self):
        """Returns smoothed 6-DOF motion + gripper + reset"""
        poses_type = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        render_poses = poses_type()
        openvr.VRCompositor().waitGetPoses(render_poses, None)
        pose = render_poses[self.device_id]

        if not pose.bPoseIsValid:
            return None

        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3]

        if self.origin_pos is None:
            self.origin_pos = pos.copy()
            return None

        # Trackpad input
        state = openvr.VRSystem().getControllerState(self.device_id)[1]
        track_x = state.rAxis[0].x
        track_y = state.rAxis[0].y

        planar = np.array([
            np.sign(track_x) * (abs(track_x)**0.8) * self.pos_scale,
            np.sign(track_y) * (abs(track_y)**0.8) * self.pos_scale,
            0.0
        ])
        z = (pos[2] - self.origin_pos[2]) * self.z_scale
        dpos = np.array([planar[1], planar[0], z])
        self.s_dpos = self._smooth_val(self.s_dpos, dpos)

        # Gripper toggle (edge detection)
        trigger_down = state.rAxis[1].x > 0.8
        if trigger_down and not self.last_trigger:
            self.gripper = -self.gripper  # toggle
        self.last_trigger = trigger_down

        # Reset button
        k_EButton_ApplicationMenu = 2
        self.reset = (state.ulButtonPressed & (1 << k_EButton_ApplicationMenu)) != 0
        if self.reset:
            self.origin_pos = pos.copy()

        return {
            "dpos": self.s_dpos.copy(),
            "drot": np.zeros(3),  # optional rotation for now
            "grasp": self.gripper,
            "reset": self.reset
        }
