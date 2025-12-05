import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="robosuite")


import robosuite
from robosuite.controllers import load_part_controller_config as load_controller_config
import numpy as np

def create_environment():
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
    return env, obs

def apply_action(env, state):
    drot = np.zeros(3)
    action = np.concatenate([state["dpos"], drot, [state["grasp"]]])
    return env.step(action)
