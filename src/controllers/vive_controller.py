import time
import numpy as np
import openvr


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-9:
        return v
    return v * (max_norm / n)


class ViveController:
    """
    Robosuite-style Vive teleop device (standalone, NOT robosuite.devices.Device)

    Behavior (stable + less drift):
      - Trackpad XY controls planar motion (with deadzone + non-linear curve)
      - Z motion is ONLY active while holding GRIP (clutch mode)
      - Motion is ONLY active while you're touching the trackpad or holding GRIP
      - Trigger toggles gripper open/close (edge-detected)
      - Menu button resets origin (re-zeros Z reference)

    Output dict matches your apply_action():
      { "dpos": np.array(3,), "grasp": float(-1 or 1), "reset": bool }
    """

    def __init__(
        self,
        device_id: int = 4,
        pos_scale: float = 0.35,     # planar gain (trackpad)
        z_scale: float = 2.0,        # vertical gain (controller delta z)
        smooth: float = 0.25,        # low-pass filter [0..1]
        deadzone_xy: float = 0.15,   # trackpad deadzone
        deadzone_z: float = 0.01,    # meters deadzone for controller z-delta
        max_step: float = 0.10,      # per-step max ||dpos|| in action units (0..1 scale)
        slow_near_limits: bool = True,
        _dbg_t = 0.0,
        _dbg_every = 0.25,
    ):
        self.device_id = device_id

        self.pos_scale = float(pos_scale)
        self.z_scale = float(z_scale)
        self.smooth = float(smooth)

        self.deadzone_xy = float(deadzone_xy)
        self.deadzone_z = float(deadzone_z)
        self.max_step = float(max_step)

        self.slow_near_limits = bool(slow_near_limits)

        self._dbg_t = _dbg_t
        self._dbg_every = _dbg_every

        # internal state
        self.origin_pos = None
        self.prev_pos = None
        self.s_dpos = np.zeros(3, dtype=np.float32)

        self.last_trigger = False
        self.gripper = -1.0
        self.reset = False

    def start_control(self, poses) -> None:
        """
        Call once after OpenVR poses are available.
        Sets origin and clears filters.
        """
        pose = poses[self.device_id]
        if not pose.bPoseIsValid:
            return

        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3].copy()

        self.origin_pos = pos.copy()
        self.prev_pos = pos.copy()
        self.s_dpos[:] = 0.0
        self.reset = False
        self.last_trigger = False
        self.gripper = -1.0

    def _smooth_val(self, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        a = self.smooth
        return (a * new) + ((1.0 - a) * prev)

    def _apply_deadzone_curve(self, x: float, dz: float, power: float = 0.85) -> float:
        """
        Deadzone + curve for nicer control:
          - inside deadzone -> 0
          - outside -> re-normalize to [0..1], then apply exponent for sensitivity
        """
        ax = abs(x)
        if ax <= dz:
            return 0.0
        # map (dz..1) -> (0..1)
        t = (ax - dz) / (1.0 - dz)
        t = np.clip(t, 0.0, 1.0)
        t = t ** power
        return float(np.sign(x) * t)

    def _button_pressed(self, state, button_id: int) -> bool:
        return (state.ulButtonPressed & (1 << button_id)) != 0

    def update(self, poses):
        pose = poses[self.device_id]
        if not pose.bPoseIsValid:
            return None

        # pose matrix -> position
        mat = np.array(list(pose.mDeviceToAbsoluteTracking), dtype=np.float32).reshape(3, 4)
        pos = mat[:, 3].copy()

        # init
        if self.origin_pos is None or self.prev_pos is None:
            self.origin_pos = pos.copy()
            self.prev_pos = pos.copy()
            return None

        # controller state
        state = openvr.VRSystem().getControllerState(self.device_id)[1]

        # --- Inputs ---
        # Trackpad axis 0 provides x,y in [-1,1]
        track_x_raw = float(state.rAxis[0].x)
        track_y_raw = float(state.rAxis[0].y)

        # Touch detection: if you're not touching trackpad, raw values often sit at 0
        # We'll treat "touch" as "magnitude above tiny threshold"
        track_active = (abs(track_x_raw) > 0.01) or (abs(track_y_raw) > 0.01)

        # Grip button (clutch for vertical)
        # Use OpenVR constants if present; fallback to common ID
        grip_id = getattr(openvr, "k_EButton_Grip", 2)
        grip_down = self._button_pressed(state, grip_id)

        # Menu button -> reset origin
        menu_id = getattr(openvr, "k_EButton_ApplicationMenu", 1)
        menu_pressed = self._button_pressed(state, menu_id)
        if menu_pressed:
            self.origin_pos = pos.copy()
            self.prev_pos = pos.copy()
            self.s_dpos[:] = 0.0
            self.reset = True
            # still return a state so the sim keeps stepping if needed
        else:
            self.reset = False

        # Trigger toggles gripper (edge detect)
        trigger_down = float(state.rAxis[1].x) > 0.80
        if trigger_down and not self.last_trigger:
            self.gripper *= -1.0
        self.last_trigger = trigger_down

        # --- Planar from trackpad (deadzone + curve) ---
        tx = self._apply_deadzone_curve(track_x_raw, self.deadzone_xy, power=0.85)
        ty = self._apply_deadzone_curve(track_y_raw, self.deadzone_xy, power=0.85)

        # IMPORTANT: you previously swapped axes to match your view (DONOT CHANGE 1 AND 0)
        # dpos = [planar_y, planar_x, z]
        planar = np.array([tx, ty], dtype=np.float32) * self.pos_scale
        dpos_planar = np.array([planar[1], planar[0], 0.0], dtype=np.float32)

        # --- Vertical from controller delta Z ONLY while gripping ---
        dz_world = float(pos[2] - self.prev_pos[2])
        self.prev_pos = pos.copy()

        if abs(dz_world) < self.deadzone_z:
            dz_world = 0.0

        dpos_z = 0.0
        if grip_down:
            # scale world meters -> action units
            dpos_z = float(dz_world * self.z_scale)

        dpos = dpos_planar + np.array([0.0, 0.0, dpos_z], dtype=np.float32)

        now = time.time()
        if now - self._dbg_t > self._dbg_every:
            self._dbg_t = now
            cls = openvr.VRSystem().getTrackedDeviceClass(self.device_id)
            # 2 == Controller class in OpenVR enum (TrackedDeviceClass_Controller)
            print(f"[VIVE] valid={pose.bPoseIsValid} class={cls} "
                  f"axis0=({track_x_raw:+.2f},{track_y_raw:+.2f}) "
                  f"track_active={track_active} grip={grip_down} "
                  f"dz={dz_world:+.4f} dpos={dpos}")
        
        mag = float(np.linalg.norm(dpos))
        if mag > 0.6:
            dpos *= 0.6 / mag

        # --- Rate limit + smoothing ---
        dpos = _clip_norm(dpos, self.max_step)
        self.s_dpos = self._smooth_val(self.s_dpos, dpos)

        return {
            "dpos": self.s_dpos.copy(),
            "grasp": float(self.gripper),
            "reset": bool(self.reset),
        }
