Structural changes:
1. All input implementation logic is moved from inline `base.py` code to `holosoma_inference/inputs` module
2. Decopuled `KeyboardListener` from `BasePolicy`.


Functional changes:
1. Unified keyboard/joystick command mappings into a single global mapping per device. So body tracking will no longer have different key assignments than locomotion. This simplifies code a lot, and the only conflicting key binding was `STAND_TOGGLE`  (moved from "start" to "back" to resolve conflict with `START_MOTION_CLIP` in wbt).
Opens:
2. (minor) `--task.use-ros` has been removed, it was not used.
3. The `"s"` key maps to velocity decrement in locomotion (via `KEYBOARD_VELOCITY_LOCOMOTION`) and to `START_MOTION_CLIP` in WBT (via `KEYBOARD_COMMANDS`).

TODOs:
