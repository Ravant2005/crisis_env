import re

with open("inference.py", "r") as f:
    orig = f.read()

patch = orig.replace(
    'phase = "coordinate"',
    'phase = "coordinate"\n            print(f"DEBUG: step={step_counter} last_coord={last_coordination_step} len={len(tracked_threats)} coord_done={coordinated_done}")'
)

with open("inference_debug.py", "w") as f:
    f.write(patch)

