# vive_controller.py
# Vive (OpenVR) controller -> robosuite OSC_POSE deltas
# Adds stable orientation control (yaw-only by default).
#
# MENU behavior:
#   - short press  : re-anchor (reset) to current EE pose
#   - long press   : request "home" (main should restore sim-start pose, then re-anchor)
#
# Returned dict includes:
#   { dpos, drot, grasp, reset, home }

import time
import numpy as np
import openvr

EPS = 1e-9


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-9:
        return v
    return v * (max_norm / n)


def _mat34_to_Rp(mat34):
    M = np.array(list(mat34), dtype=np.float64).reshape(3, 4)
    return M[:, :3].copy(), M[:, 3].copy()


def _T_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _Rp_from_T(T: np.ndarray):
    return T[:3, :3].copy(), T[:3, 3].copy()


def _rotation_vector_from_matrix(r_mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (axis*angle)."""
    R = np.asarray(r_mat, dtype=np.float64)
    trace = float(np.trace(R))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)

    axis /= axis_norm
    return (axis * theta).astype(np.float32)


def _Rz(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class ViveController:
    """
    Stable Vive teleop for robosuite OSC_POSE.

    - Translation: controller motion since clutch -> EE position target
    - Rotation:
        rotation_mode="yaw": apply only yaw component (stable)
        rotation_mode="rpy": apply full relative rotation
    - GRIP is the clutch (only moves while held)
    - MENU short press: re-anchor to current EE pose
    - MENU long press: request "home" (handled in main)
    """

    # Your observed menu bit mapping (bit index, not mask)
    MENU_BIT = 32

    # Keep SAME translation mapping:
    # robot = M @ dpos_vr
    _VRPOS_TO_ROB = np.array(
        [
            [0.0, 0.0, -1.0],  # x_robot = -z_vr
            [-1.0, 0.0, 0.0],  # y_robot = -x_vr
            [0.0, 1.0, 0.0],   # z_robot =  y_vr
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        device_id: int | None = None,

        # speed / feel
        pos_scale: float = 5.0,
        max_step_pos: float = 0.18,
        smooth: float = 0.10,
        deadzone_pos: float = 0.0006,

        # rotation
        enable_rotation: bool = True,
        rotation_mode: str = "yaw",   # "yaw" or "rpy"
        rot_scale: float = 1.25,
        max_step_rot: float = 0.35,
        deadzone_rot: float = 0.0025,

        universe: int = openvr.TrackingUniverseStanding,
        hmd_relative: bool = False,

        # menu long press threshold (sec)
        menu_hold_sec: float = 0.6,

        dbg_every: float = 0.0,
    ):
        self.device_id = device_id

        self.pos_scale = float(pos_scale)
        self.smooth = float(smooth)
        self.deadzone_pos = float(deadzone_pos)
        self.max_step_pos = float(max_step_pos)

        self.enable_rotation = bool(enable_rotation)
        self.rotation_mode = str(rotation_mode).lower().strip()
        if self.rotation_mode not in ("yaw", "rpy"):
            self.rotation_mode = "yaw"

        self.rot_scale = float(rot_scale)
        self.deadzone_rot = float(deadzone_rot)
        self.max_step_rot = float(max_step_rot)

        self.universe = universe
        self.hmd_relative = bool(hmd_relative)

        self.menu_hold_sec = float(menu_hold_sec)

        self.dbg_every = float(dbg_every)
        self._dbg_t = 0.0

        self.vrsys = None

        # clutch anchors
        self._last_grip = False
        self.ctrl_pos_at_clutch_vr = None
        self.ee_pos_at_clutch = None

        self.ctrl_R_at_clutch = None      # controller rotation at clutch start (raw VR)
        self.ee_R_at_clutch = None        # ee rotation at clutch start (robot frame)
        self.target_R = None              # target ee rotation (robot frame)

        # smoothed outputs
        self.s_dpos = np.zeros(3, dtype=np.float32)
        self.s_drot = np.zeros(3, dtype=np.float32)

        # gripper toggle
        self.last_trigger = False
        self.gripper = -1.0

        # flags
        self.reset = False
        self.home = False

        # menu press tracking (for long press)
        self._menu_down = False
        self._menu_t0 = 0.0

    # ---------------- OpenVR init + picking ----------------

    def _ensure_openvr(self):
        if self.vrsys is None:
            openvr.init(openvr.VRApplication_Other)
            self.vrsys = openvr.VRSystem()

    def _pick_controller(self) -> int | None:
        self._ensure_openvr()
        controllers = []
        right = None
        left = None
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vrsys.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_Controller:
                continue
            controllers.append(i)
            role = self.vrsys.getControllerRoleForTrackedDeviceIndex(i)
            if role == openvr.TrackedControllerRole_RightHand:
                right = i
            elif role == openvr.TrackedControllerRole_LeftHand:
                left = i
        return right if right is not None else (left if left is not None else (controllers[0] if controllers else None))

    # ---------------- Poll pose + state ----------------

    def _poll_pose_raw_vr(self):
        self._ensure_openvr()
        if self.device_id is None:
            self.device_id = self._pick_controller()
        if self.device_id is None:
            return False, np.eye(3), np.zeros(3)

        poses = self.vrsys.getDeviceToAbsoluteTrackingPose(
            self.universe, 0, openvr.k_unMaxTrackedDeviceCount
        )

        cid = int(self.device_id)
        pose_c = poses[cid]
        if not (pose_c.bDeviceIsConnected and pose_c.bPoseIsValid):
            return False, np.eye(3), np.zeros(3)

        R, p = _mat34_to_Rp(pose_c.mDeviceToAbsoluteTracking)

        if self.hmd_relative:
            hid = openvr.k_unTrackedDeviceIndex_Hmd
            pose_h = poses[hid]
            if pose_h.bDeviceIsConnected and pose_h.bPoseIsValid:
                Rh, ph = _mat34_to_Rp(pose_h.mDeviceToAbsoluteTracking)
                T_w_h = _T_from_Rp(Rh, ph)
                T_w_c = _T_from_Rp(R, p)
                T_h_c = np.linalg.inv(T_w_h) @ T_w_c
                R, p = _Rp_from_T(T_h_c)

        return True, R, p

    def _poll_state(self):
        self._ensure_openvr()
        if self.device_id is None:
            self.device_id = self._pick_controller()
        if self.device_id is None:
            return None
        try:
            _, state = self.vrsys.getControllerState(int(self.device_id))
            return state
        except Exception:
            return None

    # ---------------- Helpers ----------------

    def _reanchor_to_current(self, ee_pos: np.ndarray, ee_rotmat: np.ndarray, R_ctrl: np.ndarray, p_vr: np.ndarray):
        """Bind controller pose to current EE pose so there is no jump."""
        self.ctrl_pos_at_clutch_vr = p_vr.copy()
        self.ee_pos_at_clutch = ee_pos.copy()

        self.ctrl_R_at_clutch = R_ctrl.copy()
        self.ee_R_at_clutch = ee_rotmat.copy()
        self.target_R = ee_rotmat.copy()

        self.s_dpos[:] = 0.0
        self.s_drot[:] = 0.0
        self._last_grip = False

    # ---------------- Public API ----------------

    def start_control(self, ee_pos: np.ndarray, ee_rotmat: np.ndarray) -> None:
        ok, R_ctrl, p = self._poll_pose_raw_vr()
        if not ok:
            return

        ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        ee_rotmat = np.asarray(ee_rotmat, dtype=np.float64).reshape(3, 3)

        self._reanchor_to_current(ee_pos, ee_rotmat, R_ctrl, p)

        self.s_dpos[:] = 0.0
        self.s_drot[:] = 0.0
        self.gripper = -1.0
        self.last_trigger = False
        self.reset = False
        self.home = False

        # reset menu state
        self._menu_down = False
        self._menu_t0 = 0.0

    def update(self, ee_pos: np.ndarray, ee_rotmat: np.ndarray):
        # clear per-frame flags
        self.reset = False
        self.home = False

        ok, R_ctrl, p_vr = self._poll_pose_raw_vr()
        if not ok:
            return None

        state = self._poll_state()
        if state is None:
            return None

        ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        ee_rotmat = np.asarray(ee_rotmat, dtype=np.float64).reshape(3, 3)

        pressed = int(state.ulButtonPressed)

        # -------- MENU short/long press handling (fires on RELEASE) --------
        menu_now = (pressed & (1 << self.MENU_BIT)) != 0

        # rising edge
        if menu_now and not self._menu_down:
            self._menu_down = True
            self._menu_t0 = time.time()

        # falling edge -> decide short vs long
        if (not menu_now) and self._menu_down:
            self._menu_down = False
            held = time.time() - self._menu_t0

            if held >= self.menu_hold_sec:
                # long press => request home (main should restore sim-start pose)
                self.home = True
                # IMPORTANT: don't reanchor here; main will restore home then call start_control()
                return {"home": True}
            else:
                # short press => re-anchor to current EE pose
                self._reanchor_to_current(ee_pos, ee_rotmat, R_ctrl, p_vr)
                self.reset = True
                return None

        # -------- GRIP clutch --------
        grip_id = getattr(openvr, "k_EButton_Grip", 2)
        grip_down = (pressed & (1 << grip_id)) != 0

        # Trigger toggle
        trigger_val = float(np.clip(getattr(state.rAxis[1], "x", 0.0), 0.0, 1.0))
        trigger_down = trigger_val > 0.80
        if trigger_down and not self.last_trigger:
            self.gripper *= -1.0
        self.last_trigger = trigger_down

        # Clutch edge => anchor
        if grip_down and not self._last_grip:
            self._reanchor_to_current(ee_pos, ee_rotmat, R_ctrl, p_vr)

        self._last_grip = grip_down

        if not grip_down:
            target_dpos = np.zeros(3, dtype=np.float32)
            target_drot = np.zeros(3, dtype=np.float32)
        else:
            # ---------------- translation (unchanged) ----------------
            dpos_vr = (p_vr - self.ctrl_pos_at_clutch_vr).astype(np.float64)
            dpos_rob = (self._VRPOS_TO_ROB @ dpos_vr).astype(np.float64)

            target_pos = self.ee_pos_at_clutch + self.pos_scale * dpos_rob

            dpos = (target_pos - ee_pos).astype(np.float32)
            dpos[np.abs(dpos) < self.deadzone_pos] = 0.0
            dpos = _clip_norm(dpos, self.max_step_pos)
            target_dpos = dpos

            # ---------------- rotation (stable) ----------------
            if self.enable_rotation and self.ctrl_R_at_clutch is not None and self.ee_R_at_clutch is not None:
                # relative controller rotation since clutch
                R_rel_ctrl = R_ctrl @ self.ctrl_R_at_clutch.T

                if self.rotation_mode == "yaw":
                    # yaw from rel rotation (about +Z)
                    yaw = float(np.arctan2(R_rel_ctrl[1, 0], R_rel_ctrl[0, 0]))
                    yaw *= self.rot_scale
                    yaw = float(np.clip(yaw, -self.max_step_rot, self.max_step_rot))
                    self.target_R = _Rz(yaw) @ self.ee_R_at_clutch
                else:
                    # full orientation: apply controller rel rotation, scaled + clamped
                    rv = _rotation_vector_from_matrix(R_rel_ctrl).astype(np.float64)
                    ang = float(np.linalg.norm(rv))
                    if ang > EPS:
                        axis = rv / ang
                        ang_s = self.rot_scale * ang
                        ang_s = float(np.clip(ang_s, -self.max_step_rot, self.max_step_rot))
                        K = np.array(
                            [[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]],
                            dtype=np.float64,
                        )
                        R_scaled = np.eye(3) + np.sin(ang_s) * K + (1.0 - np.cos(ang_s)) * (K @ K)
                        self.target_R = R_scaled @ self.ee_R_at_clutch
                    else:
                        self.target_R = self.ee_R_at_clutch

                # drot to reach target from current ee
                R_err = self.target_R @ ee_rotmat.T
                drot = _rotation_vector_from_matrix(R_err).astype(np.float32)
                drot[np.abs(drot) < self.deadzone_rot] = 0.0
                drot = _clip_norm(drot, self.max_step_rot)
                target_drot = drot
            else:
                target_drot = np.zeros(3, dtype=np.float32)

        # smooth
        self.s_dpos = (self.smooth * target_dpos) + ((1.0 - self.smooth) * self.s_dpos)
        self.s_drot = (self.smooth * target_drot) + ((1.0 - self.smooth) * self.s_drot)

        # debug
        now = time.time()
        if self.dbg_every > 0 and (now - self._dbg_t) > self.dbg_every:
            self._dbg_t = now
            print(f"[VIVE] grip={int(grip_down)} dpos={self.s_dpos.round(4)} drot={self.s_drot.round(4)} grasp={self.gripper:+.0f}")

        return {
            "dpos": self.s_dpos.copy(),
            "drot": self.s_drot.copy(),
            "grasp": float(self.gripper),
            "reset": bool(self.reset),
            "home": bool(self.home),
        }