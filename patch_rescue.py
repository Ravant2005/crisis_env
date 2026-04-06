import re

with open("inference.py", "r") as f:
    text = f.read()

old_func = """            def compute_zone_priority(z):
                people = z.get("total_victims", 0) - z.get("rescued", 0)
                severity = 5.0
                accessibility = 1.0
                distance = 1.0
                return (people * 0.6) + (severity * 0.25) + (accessibility * 0.10) - (distance * 0.05)"""

new_func = """            def compute_zone_priority(z):
                # STEP 5: PRIORITY-ALIGNED RESCUE (zone_score = priority * remaining_victims)
                severity = z.get("severity", 5.0)
                pop = z.get("total_victims", 0)
                tti = max(z.get("time_to_impact", 1.0), 1.0)
                priority = (severity * pop) / tti
                remaining_victims = z.get("total_victims", 0) - z.get("rescued", 0)
                return priority * remaining_victims"""

if old_func in text:
    text = text.replace(old_func, new_func)
    with open("inference.py", "w") as f:
        f.write(text)
    print("PATCHED")
else:
    print("OLD FUNC NOT FOUND")
