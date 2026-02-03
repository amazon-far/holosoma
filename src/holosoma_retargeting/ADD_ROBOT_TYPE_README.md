## Instructions for Adding Custom Robot Type

This guide shows you how to add a new robot type (e.g., "myrobot") to the retargeting pipeline. We use T1 as example, which is already implemented.

### Overview

The process requires editing **2 main files**:
1. **`config_types/robot.py`** - Robot configuration and defaults
2. **`config_types/data_type.py`** - Joint mappings from human joints to robot joints

You'll also need to prepare robot model files (URDF/XML).

### Step 1: Prepare Robot Model Files

Create a directory for your robot model files:

```bash
mkdir -p models/myrobot
```

Place your robot files in this directory:
- **URDF file**: `models/myrobot/myrobot_{dof}dof.urdf` (e.g., `myrobot_25dof.urdf`) for viser visualization
- **XML file**: `models/myrobot/myrobot_{dof}dof.xml` for retargeting using MuJoCo

**Note**: The URDF file path follows the pattern `models/{robot_type}/{robot_type}_{dof}dof.urdf`. If your files use a different naming convention, you can override the path via command line using `--robot-config.robot-urdf-file` (e.g., `--robot-config.robot-urdf-file models/myrobot/custom_name.urdf`).

### Step 2: Add Robot Configuration in `config_types/robot.py`

Edit this file to register your robot type.

#### 2.1: Add Robot Defaults

Add your robot to the `_ROBOT_DEFAULTS` dictionary:

```python
_ROBOT_DEFAULTS: dict[str, RobotDefaults] = {
    "g1": {"robot_dof": 29, "robot_height": 1.32, "object_name": "ground"},
    "t1": {"robot_dof": 23, "robot_height": 1.2, "object_name": "ground"},
    "myrobot": {"robot_dof": 25, "robot_height": 1.4, "object_name": "ground"},  # ← Add your robot
}
```

**Parameters:**
- `robot_dof`: Number of degrees of freedom (joints) in your robot
- `robot_height`: Height of the robot in meters (used for scaling human motion)
- `object_name`: Name of the ground/object for interaction ("ground" for robot-only retargeting without object interaction)

#### 2.2: Add Robot-Specific Properties (required)

You must add these properties for your robot type:

**Foot Sticking Links** (required, used for foot-sticking constraint):
```python
def _foot_sticking_links(self) -> list[str]:
    # ... existing code ...
    if self.robot_type == "myrobot":
        return [
            "left_foot_link_1",
            "right_foot_link_1",
            # ... list all foot contact links
        ]
    raise ValueError(f"Invalid robot type: {self.robot_type}")
```

**Joint Limits** (optional, for optimization constraints - only if you need tighter limits than XML):
```python
def _manual_lb(self) -> dict[str, float]:
    # ... existing code ...
    if self.robot_type == "myrobot":
        base.update({
            "joint_index": lower_bound_value,
            # ... add joint limits for your robot
        })
    return base
```

```python
def _manual_ub(self) -> dict[str, float]:
    # ... existing code ...
    if self.robot_type == "myrobot":
        base.update({
            "joint_index": upper_bound_value,
            # ... add joint limits for your robot
        })
    return base
```

**Note**: Manual limits override specific joints that need tighter constraints. The XML file already contains joint limits for all joints, so you only need to specify manual limits for joints that need special handling beyond the XML limits (e.g., quaternion bounds for floating base, or tighter constraints for specific joints like waist or wrists). Most joints will use their limits from the XML file automatically. If your robot doesn't need any special joint limit overrides, you can skip adding robot-specific limits - the base quaternion bounds will be used automatically.

### Step 3: Add Joint Mappings in `config_types/data_type.py`

For each human motion data format you want to support, add joint mappings from human joints to your robot joints.

#### 3.1: Add Joint Mappings

Add entries to `JOINTS_MAPPINGS` for each `(data_format, robot_type)` combination you want to support:

```python
JOINTS_MAPPINGS = {
    # ... existing mappings ...

    # For LAFAN data format with your robot
    ("lafan", "myrobot"): {
        "Spine1": "base_link",              # Human joint → Robot joint
        "LeftUpLeg": "left_hip_joint",
        "RightUpLeg": "right_hip_joint",
        "LeftLeg": "left_knee_joint",
        "RightLeg": "right_knee_joint",
        "LeftArm": "left_shoulder_joint",
        "RightArm": "right_shoulder_joint",
        "LeftForeArm": "left_elbow_joint",
        "RightForeArm": "right_elbow_joint",
        "LeftFoot": "left_ankle_joint",
        "RightFoot": "right_ankle_joint",
        "LeftToeBase": "left_toe_joint",
        "RightToeBase": "right_toe_joint",
        "LeftHand": "left_hand_joint",
        "RightHand": "right_hand_joint",
    },

    # For SMPLH data format with your robot
    ("smplh", "myrobot"): {
        "Pelvis": "base_link",
        "L_Hip": "left_hip_joint",
        "R_Hip": "right_hip_joint",
        # ... map all relevant joints
    },

    # Add mappings for other data formats (smplx, mocap) as needed
}
```

**Key Points:**
- The **key** (left side) is the human joint name from the motion data format
- The **value** (right side) is the corresponding robot joint/link name from your URDF
- You only need to map joints that are relevant for retargeting (typically: pelvis, hips, knees, ankles, shoulders, elbows, wrists)

### Summary: What You Need to Edit

**In `config_types/robot.py`:**
1. ✅ Add entry to `_ROBOT_DEFAULTS` dictionary (required)
2. ✅ Add `_foot_sticking_links()` case (required)
3. ⚠️ Add `_manual_lb()` / `_manual_ub()` cases (optional, only if you need tighter limits than XML)

**In `config_types/data_type.py`:**
7. ✅ Add joint mappings to `JOINTS_MAPPINGS` for each `(data_format, "myrobot")` combination (required)

**File System:**
8. ✅ Create `models/myrobot/` directory
9. ✅ Place URDF file: `models/myrobot/myrobot_{dof}dof.urdf`
10. ✅ Place XML file: `models/myrobot/myrobot_{dof}dof.xml`

### Ready to Run

Once configured, you can use your custom robot:

```bash
python examples/robot_retarget.py \
  --data_path /path/to/data \
  --task-type robot_only \
  --task-name your_sequence \
  --data_format smplh \
  --retargeter.debug \
  --retargeter.visualize \
  --robot myrobot
```
