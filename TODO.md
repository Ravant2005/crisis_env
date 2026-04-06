# Fix Worker Rollout TypeError: int() dict

## Plan Steps

### 1. Create this TODO.md ✅
### 2. Guard utils.py threat/resource/zone ID extractions ✅
### 3. Fix policy_model.py decode_action threat_ids validation ✅
### 4. Enforce server/environment.py threat_id always int ✅
### 5. Add train.py forced_action ID checks + debug logging
### 6. Test: run `python train.py` → verify no crashes, rollouts succeed
### 7. Update TODO.md mark complete
### 8. Run `python evaluate.py` validate performance

**Goal:** Eliminate `[WARN] Worker rollout failed` → full episodes, improved baseline scores.

Current Issue: Malformed obs.threats threat_id=dict → int(dict) crash in parallel env.reset.

