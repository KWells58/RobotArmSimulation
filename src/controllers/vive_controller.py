import time
import numpy as np
import openvr


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-9:
        return v
    return v * (max_norm / n)


def _rotation_vector_from_matrix(r_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector (axis * angle)."""
    trace = float(np.trace(r_mat))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)

    axis = np.array(
        [
            r_mat[2, 1] - r_mat[1, 2],
            r_mat[0, 2] - r_mat[2, 0],
            r_mat[1, 0] - r_mat[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)

    axis /= axis_norm
    return (axis * theta).astype(np.float32)


class ViveController:
    """
    Robosuite-style Vive teleop device (standalone, NOT robosuite.devices.Device).

    Behavior:
      - Hold GRIP for motion clutch (safe enable)
      - While clutch is held:
          - controller translation drives end-effector dpos (3DOF)
          - controller orientation delta drives end-effector drot (3DOF)
      - Trigger toggles gripper open/close (edge-detected)
      - Menu button resets motion reference origin
    """

    def __init__(
        self,
        device_id: int = 4,
        pos_scale: float = 3.0,
        rot_scale: float = 2.0,
        smooth: float = 0.25,
        deadzone_pos: float = 0.0015,
        deadzone_rot: float = 0.004,
        max_step_pos: float = 0.12,
        max_step_rot: float = 0.24,
        _dbg_t=0.0,
        _dbg_every=0.25,
    ):
        self.device_id = device_id

        self.pos_scale = float(pos_scale)
        self.rot_scale = float(rot_scale)
        self.smooth = float(smooth)

        self.deadzone_pos = float(deadzone_pos)
        self.deadzone_rot = float(deadzone_rot)
        self.max_step_pos = float(max_step_pos)
        self.max_step_rot = float(max_step_rot)

        self._dbg_t = _dbg_t
        self._dbg_every = _dbg_every

        self.origin_pos = None
        self.prev_pos = None
        self.prev_rot = None

        self.s_dpos = np.zeros(3, dtype=np.float32)
        self.s_drot = np.zeros(3, dtype=np.float32)

        self.last_trigger = False
        self.gripper = -1.0
        self.reset = False

    def start_control(self, poses) -> None:
        pose = poses[self.device_id]
        if not pose.bPoseIsValid:
            return

        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3].copy()
        rot = mat[:, :3].copy()

        self.origin_pos = pos.copy()
        self.prev_pos = pos.copy()
        self.prev_rot = rot.copy()
        self.s_dpos[:] = 0.0
        self.s_drot[:] = 0.0
        self.reset = False
        self.last_trigger = False
        self.gripper = -1.0

    def _smooth_val(self, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        a = self.smooth
        return (a * new) + ((1.0 - a) * prev)

    def _button_pressed(self, state, button_id: int) -> bool:
        return (state.ulButtonPressed & (1 << button_id)) != 0

    def update(self, poses):
        pose = poses[self.device_id]
        if not pose.bPoseIsValid:
            return None

        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3].copy()
        rot = mat[:, :3].copy()

        if self.origin_pos is None or self.prev_pos is None or self.prev_rot is None:
            self.origin_pos = pos.copy()
            self.prev_pos = pos.copy()
            self.prev_rot = rot.copy()
            return None

        state = openvr.VRSystem().getControllerState(self.device_id)[1]

        grip_id = getattr(openvr, "k_EButton_Grip", 2)
        grip_down = self._button_pressed(state, grip_id)

        menu_id = getattr(openvr, "k_EButton_ApplicationMenu", 1)
        menu_pressed = self._button_pressed(state, menu_id)
        if menu_pressed:
            self.origin_pos = pos.copy()
            self.prev_pos = pos.copy()
            self.prev_rot = rot.copy()
            self.s_dpos[:] = 0.0
            self.s_drot[:] = 0.0
            self.reset = True
        else:
            self.reset = False

        trigger_down = float(state.rAxis[1].x) > 0.80
        if trigger_down and not self.last_trigger:
            self.gripper *= -1.0
        self.last_trigger = trigger_down

        if not grip_down:
            self.prev_pos = pos.copy()
            self.prev_rot = rot.copy()
            target_dpos = np.zeros(3, dtype=np.float32)
            target_drot = np.zeros(3, dtype=np.float32)
        else:
            dpos_world = (pos - self.prev_pos).astype(np.float32)
            self.prev_pos = pos.copy()

            rel_rot = rot @ self.prev_rot.T
            drot_world = _rotation_vector_from_matrix(rel_rot)
            self.prev_rot = rot.copy()

            dpos_world[np.abs(dpos_world) < self.deadzone_pos] = 0.0
            drot_world[np.abs(drot_world) < self.deadzone_rot] = 0.0

            target_dpos = dpos_world * self.pos_scale
            target_drot = drot_world * self.rot_scale

            target_dpos = _clip_norm(target_dpos, self.max_step_pos)
            target_drot = _clip_norm(target_drot, self.max_step_rot)

        self.s_dpos = self._smooth_val(self.s_dpos, target_dpos)
        self.s_drot = self._smooth_val(self.s_drot, target_drot)

        now = time.time()
        if now - self._dbg_t > self._dbg_every:
            self._dbg_t = now
            cls = openvr.VRSystem().getTrackedDeviceClass(self.device_id)
            print(
                f"[VIVE] valid={pose.bPoseIsValid} class={cls} clutch={grip_down} "
                f"dpos={self.s_dpos} drot={self.s_drot}"
            )

        return {
            "dpos": self.s_dpos.copy(),
            "drot": self.s_drot.copy(),
            "grasp": float(self.gripper),
            "reset": bool(self.reset),
        }
