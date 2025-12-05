import robosuite
from robosuite.controllers import load_composite_controller_config
import numpy as np

def create_environment():
    env = robosuite.make(
        "Lift",
        robots=["UR5e"],
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
    return env, obs

def apply_action(env, state):
    drot = np.zeros(3)
    action = np.concatenate([state["dpos"], drot, [state["grasp"]]])
    return env.step(action)
