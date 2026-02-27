import robosuite
import numpy as np


def create_environment(
    env_name="Lift",
    robots=("UR5e",),
    control_freq=20,
):
    """
    Create a robosuite environment.

    For single-arm:
        env_name="Lift", robots=("UR5e",)

    For dual-arm / bimanual (if available in your robosuite):
        env_name="TwoArmLift", robots=("Panda", "Panda")   (example)
        OR Baxter-style if your env supports it.

    """
    env = robosuite.make(
        env_name,
        robots=list(robots),
        gripper_types="default",
        has_renderer=True,
        render_camera="frontview",
        control_freq=control_freq,
        ignore_done=True,
        horizon=10000,
        use_object_obs=False,
        use_camera_obs=False,
    )
    obs = env.reset()
    return env, obs


def state_to_osc_action(state: dict) -> np.ndarray:
    """
    Convert ViveController state dict -> 7D OSC_POSE-style action:
      [dpos(3), drot(3), grasp(1)]
    """
    dpos = np.asarray(state["dpos"], dtype=np.float32).reshape(3)
    drot = np.asarray(state.get("drot", np.zeros(3)), dtype=np.float32).reshape(3)
    grasp = np.asarray([state["grasp"]], dtype=np.float32)
    return np.concatenate([dpos, drot, grasp], dtype=np.float32)


def apply_action(env, state_right: dict | None, state_left: dict | None = None):
    """
    Apply either single-arm or dual-arm action.

    If env.action_dim <= 7:
        uses state_right only
    Else:
        packs [right_arm(7), left_arm(7)] by default.
        If your env expects [left, right], just swap in the code below.

    Returns: obs, reward, done, info
    """
    adim = int(getattr(env, "action_dim", 0))

    if adim <= 7:
        if state_right is None:
            action = np.zeros(adim, dtype=np.float32)
        else:
            a = state_to_osc_action(state_right)
            action = a[:adim] if a.size >= adim else np.pad(a, (0, adim - a.size))
        return env.step(action)

    # Dual-arm expected action size is usually 14 (2 * 7)
    aR = np.zeros(7, dtype=np.float32) if state_right is None else state_to_osc_action(state_right)
    aL = np.zeros(7, dtype=np.float32) if state_left is None else state_to_osc_action(state_left)

    # Default packing: [RIGHT, LEFT]
    action = np.concatenate([aL, aR], dtype=np.float32)

    # If your env wants [LEFT, RIGHT], use this instead:
    # action = np.concatenate([aL, aR], dtype=np.float32)

    if action.size < adim:
        action = np.pad(action, (0, adim - action.size))
    else:
        action = action[:adim]

    return env.step(action)