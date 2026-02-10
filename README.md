# RobotArmSimulation

A Python-based teleoperation system for a UR5e robot arm in a robosuite simulation, using HTC Vive controllers for intuitive control. The latest control mode maps the Vive controller's full 6-DOF motion (position + orientation) directly to the robot end-effector while using grip as a safety clutch.

## Features

- Real-time teleoperation of UR5e in the Lift environment
- 6-DOF end-effector control from Vive controller motion
- Safety clutch (hold grip to move the robot)
- Gripper toggle using the trigger button
- Reset motion origin using the menu button
- Smoothed and deadzoned motion for more stable teleoperation
- Modular Python project for easy expansion

## Folder Structure

```text
RobotArmSimulation/
├─ src/
│  ├─ main.py
│  ├─ simulation/
│  │  ├─ __init__.py
│  │  └─ robo_env.py
│  ├─ controllers/
│  │  ├─ __init__.py
│  │  └─ vive_controller.py
│  └─ helpers/
│     └─ detect.py
├─ README.md
└─ requirements.txt
```

## Requirements

- Python 3.12 (or compatible 3.x)
- pip
- robosuite
- openvr
- numpy
- Mujoco 2.x installed and activated
- HTC Vive Pro 2 and controllers
- SteamVR running and tracking healthy
- Compatible GPU for rendering (OpenGL)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KWells58/RobotArmSimulation.git
cd RobotArmSimulation
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

- Windows PowerShell: `./venv/Scripts/Activate.ps1`
- Windows CMD: `./venv/Scripts/activate.bat`
- Linux/macOS: `source venv/bin/activate`

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set up robosuite macros:

```bash
python venv/Lib/site-packages/robosuite/scripts/setup_macros.py
```

## Usage

1. Ensure Vive controllers are charged and SteamVR is running.
2. Activate the virtual environment.
3. Run teleoperation:

```bash
python -m src.main
```

### Controls

- Hold **Grip** → enable motion clutch (required for movement)
- Controller translation (while gripping) → end-effector XYZ motion
- Controller rotation (while gripping) → end-effector orientation motion
- **Trigger** → toggle gripper open/close
- **Menu** → reset tracking origin and motion filters

### Tuning Notes

If motion feels too sensitive or too slow, tune these parameters in `ViveController(...)`:

- `pos_scale`
- `rot_scale`
- `deadzone_pos`
- `deadzone_rot`
- `max_step_pos`
- `max_step_rot`
- `smooth`

## Troubleshooting

- Run `python -m src.helpers.detect` to verify controller IDs and roles.
- If no controller is found, check SteamVR status and USB/Bluetooth pairing.
- If motion jitters, increase deadzones slightly and reduce scale.

## Contributing

- Add controller features, IK improvements, or robot environments.
- Keep device logic in `src/controllers` and simulation wiring in `src/simulation`.

## License

Add a license file if you plan to distribute this project.
