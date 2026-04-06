import re

with open("inference.py", "r") as f:
    orig = f.read()

patch = orig.replace(
    'elif not coordinated_done or (coordinated_done and len(tracked_threats) >= 2 and (step_counter - last_coordination_step >= 5)):',
    'elif not coordinated_done or (coordinated_done and len(tracked_threats) >= 2 and (step_counter - last_coordination_step >= 5) and _live_resource_budget > 0):'
)

with open("inference.py", "w") as f:
    f.write(patch)

