import re

with open("inference.py", "r") as f:
    orig = f.read()

start_marker = "    # ── Track which tasks have been done this episode ──────────────────────"
end_marker = "    # ── Final scores ───────────────────────────────────────────────────────"

new_block = """    # ── Track which tasks have been done this episode ──────────────────────
    classified: set = set()
    predicted: set = set()
    allocated: set = set()
    coordinated_done: bool = False
    allocation_count: int = 0
    last_coordination_step: int = -999

    # ── Episode loop ───────────────────────────────────────────────────────
    step_counter = 0
    _live_resource_budget = 99  # initialized; updated after every http_step call
    
    while not done and step_counter < 50:
        step_counter += 1
        
        # Get current active threats
        active_threats = [t for t in threats if t.get("status") == "active"]
        tracked_threats = list(threats)
        active_ids = {t["threat_id"] for t in active_threats}
        
        classified &= active_ids
        predicted &= active_ids
        allocated &= active_ids

        unclassified = [t for t in active_threats if t["threat_id"] not in classified]
        unpredicted = [t for t in active_threats if t["threat_id"] not in predicted]

        # ── PHASE DECISION ENGINE ──────────────────────────────────────────
        phase = None
        if unclassified:
            phase = "classify"
        elif unpredicted:
            phase = "predict"
        elif not coordinated_done or (coordinated_done and len(tracked_threats) >= 2 and (step_counter - last_coordination_step >= 5)):
            phase = "coordinate"
        elif len(allocated) < min(len(active_threats), _live_resource_budget):
            phase = "allocate"
        else:
            phase = "rescue"

        # Collect actions to execute this step (max MAX_ACTIONS_PER_STEP)
        actions_to_execute: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

        if phase == "classify":
            for threat in unclassified:
                actions_to_execute.append(("classify", threat, _classify_action(threat)))

        elif phase == "predict":
            for threat in unpredicted:
                actions_to_execute.append(("predict", threat, _predict_action(threat)))

        elif phase == "coordinate":
            # Early coordinate immediately after predict
            coord_scope = active_threats if len(active_threats) >= 2 else tracked_threats
            if len(coord_scope) >= 2:
                if llm_client:
                    order = _llm_suggest_priority(coord_scope, llm_client)
                    coord_action = {
                        "action_type":  "coordinate",
                        "coordination": {"priority_order": order},
                    }
                else:
                    ranked = sorted(coord_scope, key=_priority_score, reverse=True)
                    coord_action = {
                        "action_type": "coordinate",
                        "coordination": {"priority_order": [t["threat_id"] for t in ranked]},
                    }
                actions_to_execute.append(("coordinate", {"threat_id": "all"}, coord_action))
            else:
                # If only 1 threat, skip coordinate and move to allocate
                phase = "allocate"

        if phase == "allocate":
            ranked_threats = sorted(active_threats, key=_priority_score, reverse=True)
            budget_remaining = _live_resource_budget
            max_allocations = min(len(ranked_threats), _live_resource_budget)
            
            for threat in ranked_threats:
                if budget_remaining <= 0:
                    break
                if len(allocated) + len(actions_to_execute) >= max_allocations:
                    break
                tid = threat["threat_id"]
                if tid not in allocated and threat.get("assigned_resource") is None:
                    alloc = _allocate_action(threat, resources)
                    if alloc:
                        actions_to_execute.append(("allocate", threat, alloc))
                        budget_remaining -= 1
                        for r in resources:
                            if r["resource_id"] == alloc["allocation"]["resource_id"]:
                                r["is_available"] = False
                                break

        elif phase == "rescue":
            global current_rescue_target
            if "current_rescue_target" not in globals():
                current_rescue_target = None

            current_impacted_zones = [z for z in zones if z.get("is_active", False)]
            
            # Highest priority zones first
            MIN_PEOPLE_THRESHOLD = 5
            active_zones = [
                z for z in current_impacted_zones
                if (z.get("total_victims", 0) - z.get("rescued", 0)) >= MIN_PEOPLE_THRESHOLD
            ]
            
            if active_zones:
                target_zone = max(active_zones, key=lambda z: z.get("total_victims", 0) - z.get("rescued", 0))
                current_rescue_target = target_zone["zone_id"]
                
                people_remaining = target_zone.get("total_victims", 0) - target_zone.get("rescued", 0)
                _live_budget = _live_resource_budget
                
                while _live_budget >= 1 and people_remaining >= MIN_PEOPLE_THRESHOLD:
                    rescue_act = _rescue_action(target_zone, budget_remaining=_live_budget)
                    if rescue_act:
                        actions_to_execute.append(("rescue", target_zone, rescue_act))
                        units_sent = rescue_act["rescue"]["rescue_units_to_send"]
                        _live_budget -= units_sent
                        people_remaining -= (units_sent * 15)
                    else:
                        break

        # ── Execute collected actions (up to MAX_ACTIONS_PER_STEP) ─────────
        executed = 0
        
        for action_label, target_obj, action_payload in actions_to_execute[:MAX_ACTIONS_PER_STEP]:
            executed += 1
            
            target_label = (
                f"threat_{target_obj.get('threat_id', '?')}"
                if "threat_id" in target_obj
                else f"zone_{target_obj.get('zone_id', '?')}"
            )
            
            # Determine decision reasoning
            if action_label == "classify":
                reasoning = "classify threat to understand its characteristics"
            elif action_label == "predict":
                reasoning = "predict TTI and population for accurate resource allocation"
            elif action_label == "coordinate":
                reasoning = "set priority order for all active threats"
            elif action_label == "allocate":
                pref = _preferred_action(target_obj)
                reasoning = f"assign best resource to high-priority threat ({pref})"
            elif action_label == "rescue":
                reasoning = "deploy rescue units to save victims in impacted zone"
            else:
                reasoning = "taking action on critical threat"
            
            # Execute action with session_id
            result = http_step(action_payload, session_id)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs_data = result.get("observation", {})
            alerts = obs_data.get("alerts", [])
            
            # Update local state from fresh observation
            threats    = obs_data.get("threats",        threats)
            resources  = obs_data.get("resources",      resources)
            zones      = obs_data.get("affected_zones", zones)
            # Track live budget — used by rescue and allocation sizing
            _live_resource_budget = int(obs_data.get("resource_budget_remaining", 99))
            
            # Get cumulative score with session_id
            state = http_state(session_id)
            cumulative = state.get("final_score", 0.0)
            
            result_label = alerts[0][:80] if alerts else "ok"
            log_step(action_label, target_label, result_label, reward, done, reasoning, cumulative)
            
            if done:
                break
        
        # After executing actions, update tracking sets for successfully completed actions
        for action_label, target_obj, action_payload in actions_to_execute[:executed]:
            if action_label == "classify":
                tid = action_payload.get("classification", {}).get("threat_id")
                if tid is not None:
                    classified.add(int(tid))
            if action_label == "predict":
                tid = action_payload.get("prediction", {}).get("threat_id")
                if tid is not None:
                    predicted.add(int(tid))
            if action_label == "allocate":
                alloc_tid = action_payload.get("allocation", {}).get("threat_id")
                if alloc_tid is not None:
                    allocated.add(int(alloc_tid))
                    allocation_count += 1
            if action_label == "coordinate":
                coordinated_done = True
                last_coordination_step = step_counter
        
        # Small delay between steps
        time.sleep(STEP_DELAY)

        if done:
            break

"""

start_idx = orig.find(start_marker)
end_idx = orig.find(end_marker)

if start_idx != -1 and end_idx != -1:
    patched = orig[:start_idx] + new_block + orig[end_idx:]
    with open("inference.py", "w") as f:
        f.write(patched)
    print("SUCCESS")
else:
    print("FAILED TO FIND MARKERS")

