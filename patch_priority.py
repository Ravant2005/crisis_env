import re

with open("inference.py", "r") as f:
    text = f.read()

# Add last_coord_order init
old_init = """    coordinated = False
    last_coord_step = -1"""
new_init = """    coordinated = False
    last_coord_step = -1
    last_coord_order = []"""
text = text.replace(old_init, new_init)

# Modify phase decision logic to use priorities_changed
old_phase = """        elif not coordinated or (coordinated and len(tracked_threats) >= 2 and (step - last_coord_step >= 5) and _live_resource_budget > 0):
            phase = "coordinate"
        elif len(allocated) < min(len(active_threats), _live_resource_budget):"""
new_phase = """        elif not coordinated:
            phase = "coordinate"
        elif coordinated and len(tracked_threats) >= 2 and (step - last_coord_step >= 5):
            # Determine if priorities changed
            scope = active_threats if len(active_threats) >= 2 else tracked_threats
            ranked = sorted(scope, key=_priority_score, reverse=True)
            current_order = [t["threat_id"] for t in ranked]
            if current_order != last_coord_order:
                phase = "coordinate"
            elif len(allocated) < min(len(active_threats), _live_resource_budget):
                phase = "allocate"
            else:
                phase = "rescue"
        elif len(allocated) < min(len(active_threats), _live_resource_budget):"""
text = text.replace(old_phase, new_phase)

# Update last_coord_order when coordinating
old_coord_append = """                else:
                    ranked = sorted(coord_scope, key=_priority_score, reverse=True)
                    coord_action = {
                        "action_type": "coordinate",
                        "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
                    }
                actions_to_execute.append(("coordinate", {"threat_id": "all"}, coord_action))"""
new_coord_append = """                else:
                    ranked = sorted(coord_scope, key=_priority_score, reverse=True)
                    new_order = [t["threat_id"] for t in ranked]
                    coord_action = {
                        "action_type": "coordinate",
                        "coordination": {"priority_order": new_order},
                    }
                actions_to_execute.append(("coordinate", {"threat_id": "all"}, coord_action))"""
text = text.replace(old_coord_append, new_coord_append)

# Update the tracker
old_update = """            if action_label == "coordinate":
                coordinated = True
                last_coord_step = step"""
new_update = """            if action_label == "coordinate":
                coordinated = True
                last_coord_step = step
                last_coord_order = action_payload.get("coordination", {}).get("priority_order", [])"""
text = text.replace(old_update, new_update)

with open("inference.py", "w") as f:
    f.write(text)

print("PATCH APPLIED")
