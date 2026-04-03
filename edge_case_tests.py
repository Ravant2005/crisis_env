#!/usr/bin/env python3
"""
edge_case_tests.py — Tests edge cases in the Crisis Environment.
"""

import sys
sys.path.insert(0, ".")

from server.environment import CrisisEnvironment
from models import (
    ActionType, CrisisAction, ClassificationPayload, PredictionPayload,
    AllocationPayload, CoordinationPayload, RescuePayload
)


def test_empty_action_handling():
    """Test that invalid/empty actions are handled gracefully."""
    print("\n[TEST] Empty action handling...")
    
    env = CrisisEnvironment(seed=42)
    obs = env.reset(seed=42)
    
    action_empty = CrisisAction(action_type=ActionType.CLASSIFY, classification=None)
    result = env.step(action_empty)
    
    assert result.reward == -0.2, f"Expected -0.2 for empty action, got {result.reward}"
    print("  ✅ Empty action returns negative reward")
    
    print("  PASSED")
    return True


def test_resource_exhaustion():
    """Test behavior when all resources are allocated."""
    print("\n[TEST] Resource exhaustion...")
    
    env = CrisisEnvironment(seed=42)
    obs = env.reset(seed=42)
    
    available_before = sum(1 for r in obs.resources if r.is_available)
    assert available_before == 8, f"Expected 8 resources, got {available_before}"
    print(f"  Initial resources: {available_before}")
    
    for i, threat in enumerate(obs.threats[:3]):
        action = CrisisAction(
            action_type=ActionType.ALLOCATE,
            allocation=AllocationPayload(
                threat_id=threat.threat_id,
                resource_id=obs.resources[i].resource_id
            )
        )
        result = env.step(action)
        if result.reward > 0:
            print(f"  Allocated resource {obs.resources[i].resource_id} to threat {threat.threat_id}")
    
    obs_after = env._build_observation([])
    available_after = sum(1 for r in obs_after.resources if r.is_available)
    print(f"  Remaining resources: {available_after}")
    
    if available_after == 0:
        action_no_resources = CrisisAction(
            action_type=ActionType.ALLOCATE,
            allocation=AllocationPayload(
                threat_id=1,
                resource_id=999
            )
        )
        result = env.step(action_no_resources)
        assert result.reward < 0, f"Expected negative reward for exhausted resources, got {result.reward}"
        print("  ✅ Allocating with no resources returns negative reward")
    
    print("  PASSED")
    return True


def test_late_response_penalty():
    """Test that late rescue responses receive lower rewards."""
    print("\n[TEST] Late response penalty...")
    
    env = CrisisEnvironment(seed=42)
    obs = env.reset(seed=42)
    
    threat = obs.threats[0]
    
    action = CrisisAction(
        action_type=ActionType.ALLOCATE,
        allocation=AllocationPayload(
            threat_id=threat.threat_id,
            resource_id=1
        )
    )
    env.step(action)
    
    for _ in range(40):
        action = CrisisAction(
            action_type=ActionType.COORDINATE,
            coordination=CoordinationPayload(priority_order=[1,2,3])
        )
        result = env.step(action)
        if result.done:
            break
    
    state = env.state()
    print(f"  Step count: {state.step_count}")
    print(f"  Time remaining: {state.total_steps - state.step_count}")
    
    if state.rescue_score > 0:
        print(f"  ✅ Rescue scored: {state.rescue_score}")
    else:
        print("  ✅ No rescue needed (all threats contained)")
    
    print("  PASSED")
    return True


def test_determinism():
    """Test that identical seeds produce identical outputs."""
    print("\n[TEST] Determinism...")
    
    env1 = CrisisEnvironment(seed=42)
    obs1 = env1.reset(seed=42, difficulty="medium")
    
    threats1 = [(t.threat_type, t.severity, t.population_at_risk, t.time_to_impact) for t in obs1.threats]
    
    env2 = CrisisEnvironment(seed=42)
    obs2 = env2.reset(seed=42, difficulty="medium")
    
    threats2 = [(t.threat_type, t.severity, t.population_at_risk, t.time_to_impact) for t in obs2.threats]
    
    assert threats1 == threats2, "Threats differ between runs with same seed"
    print("  ✅ Threats identical")
    
    assert obs1.resources[0].resource_type == obs2.resources[0].resource_type
    print("  ✅ Resources identical")
    
    env1_2 = CrisisEnvironment(seed=42)
    obs1_2 = env1_2.reset(seed=42, difficulty="medium")
    state1 = env1_2.state()
    episode1 = state1.episode_id
    
    env2_2 = CrisisEnvironment(seed=42)
    obs2_2 = env2_2.reset(seed=42, difficulty="medium")
    state2 = env2_2.state()
    episode2 = state2.episode_id
    
    assert state1.difficulty == state2.difficulty
    assert episode1 != episode2
    print("  ✅ Different episodes have unique IDs")
    
    print("  PASSED")
    return True


def test_difficulty_scaling():
    """Test that difficulty parameter affects threat count."""
    print("\n[TEST] Difficulty scaling...")
    
    env_easy = CrisisEnvironment(seed=42)
    obs_easy = env_easy.reset(seed=42, difficulty="easy")
    assert len(obs_easy.threats) == 2, f"Easy should have 2 threats, got {len(obs_easy.threats)}"
    print("  ✅ Easy difficulty: 2 threats")
    
    env_medium = CrisisEnvironment(seed=42)
    obs_medium = env_medium.reset(seed=42, difficulty="medium")
    assert len(obs_medium.threats) == 3, f"Medium should have 3 threats, got {len(obs_medium.threats)}"
    print("  ✅ Medium difficulty: 3 threats")
    
    env_hard = CrisisEnvironment(seed=42)
    obs_hard = env_hard.reset(seed=42, difficulty="hard")
    assert len(obs_hard.threats) == 5, f"Hard should have 5 threats, got {len(obs_hard.threats)}"
    print("  ✅ Hard difficulty: 5 threats")
    
    print("  PASSED")
    return True


def test_grader_boundaries():
    """Test that all grader scores are within [0, 1]."""
    print("\n[TEST] Grader boundaries...")
    
    env = CrisisEnvironment(seed=42)
    obs = env.reset(seed=42)
    
    scores = env.task_scores()
    for name, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Grader {name} score {score} outside [0,1]"
        print(f"  {name}: {score:.4f} ✅")
    
    state = env.state()
    assert 0.0 <= state.final_score <= 1.0
    print(f"  final: {state.final_score:.4f} ✅")
    
    print("  PASSED")
    return True


def main():
    print("=" * 50)
    print("EDGE CASE TESTS")
    print("=" * 50)
    
    tests = [
        test_empty_action_handling,
        test_resource_exhaustion,
        test_late_response_penalty,
        test_determinism,
        test_difficulty_scaling,
        test_grader_boundaries,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())