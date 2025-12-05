# RobotArmSimulation

A Python-based teleoperation system for a UR5e robot arm in a robosuite simulation, using HTC Vive controllers for intuitive control. The system maps Vive controller inputs to planar movement, vertical movement, and gripper actions.

## Features

- Real-time teleoperation of UR5e in the Lift environment
- Planar movement controlled via Vive trackpad
- Vertical movement based on controller position
- Gripper toggle using the trigger button
- Reset origin functionality using the menu button
- Smoothed motion for more natural control
- Modular Python project for easy expansion

## Folder Structure
```
RobotArmSimulation/
├─ src/
│  ├─ main.py
|  ├─ simulation/
|  |  ├─ __init__.py
|  |  └─ robo_env.py
│  └─ controllers/
│     ├─ __init__.py
│     └─ vive_controller.py
├─ venv/  # Python virtual environment
├─ README.md
└─ requirements.txt  # Project dependencies
```

## Requirements

- Python 3.12 (or compatible 3.x)
- pip
- robosuite
- openvr
- numpy
- Mujoco 2.x installed and activated
- HTC Vive pro 2 and controllers
- Compatible GPU for rendering (OpenGL)

## Installation

1. Clone the repository:

git clone https://github.com/KWells58/RobotArmSimulation.git
cd RobotArmSimulation

2. Create a virtual environment:

python -m venv venv

3. Activate the virtual environment:

- Windows PowerShell: `.\venv\Scripts\Activate.ps1`
- Windows CMD: `.\venv\Scripts\activate.bat`
- Linux/macOS: `source venv/bin/activate`

4. Install dependencies:

pip install -r requirements.txt

5. Install and configure Mujoco:

- Download Mujoco 2.x from the Mujoco website
- Place `mujoco.dll` (Windows) or `libmujoco.so` (Linux) in a location accessible by Python or update your PATH/LD_LIBRARY_PATH
- Test Mujoco installation with `import mujoco` in Python

6. Set up robosuite macros:

python venv/Lib/site-packages/robosuite/scripts/setup_macros.py

## Usage

1. Ensure Vive controllers are fully charged and SteamVR is running.
2. Activate the virtual environment.
3. Run the teleoperation script:

python -m src.main
### Controls

- Trackpad XY → Planar movement (X/Y)
- Controller Z → Vertical movement
- Trigger → Toggle gripper
- Menu button → Reset origin

### Notes

- Wrist rotation is currently disabled to isolate planar movement
- Planar and vertical movement are smoothed for more natural control
- Debug messages can be enabled inside `vive_controller.py` for troubleshooting

## Contributing

- Add new controller features, IK improvements, or robot environments
- Use modular structure in `src/controllers` for additional devices or input mappings

## License
