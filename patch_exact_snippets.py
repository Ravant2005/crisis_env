import re

with open("inference.py", "r") as f:
    text = f.read()

# Replace variables
text = text.replace("coordinated_done: bool = False", "coordinated = False")
text = text.replace("last_coordination_step: int = -999", "last_coord_step = -1")
text = text.replace("last_coordination_step", "last_coord_step")
text = text.replace("coordinated_done", "coordinated")
text = text.replace("step_counter", "step")

with open("inference.py", "w") as f:
    f.write(text)

