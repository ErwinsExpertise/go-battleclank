#!/usr/bin/env python3
"""
Integration test to demonstrate the fix for death_analysis.by_reason

This test simulates the flow from run_benchmark.py through continuous_training.py
to verify that death reasons and phases are properly separated.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_benchmark import BenchmarkRunner


def test_death_analysis_integration():
    """Test the complete flow of death analysis from benchmark to training"""
    
    print("="*70)
    print("Integration Test: Death Analysis Fix")
    print("="*70)
    
    # Simulate a benchmark run with various death types
    runner = BenchmarkRunner(num_games=10)
    
    # Simulate game results with different death scenarios
    test_scenarios = [
        # (turns, winner, output) - simulating different death types
        (75, "rust-baseline", "Game completed. go-battleclank eliminated by head collision."),
        (120, "rust-baseline", "Game completed. go-battleclank eliminated by starvation."),
        (30, "rust-baseline", "Game completed. go-battleclank eliminated by self collision."),
        (180, "rust-baseline", "Game completed. go-battleclank eliminated by body collision."),
        (45, "rust-baseline", "Game completed. go-battleclank eliminated by head collision."),
        (160, "rust-baseline", "Game completed. go-battleclank eliminated by starvation."),
        (90, "rust-baseline", "Game completed. go-battleclank eliminated by wall collision."),
    ]
    
    # Aggregate results as run_benchmark.py would do
    death_reasons = {}
    death_by_phase = {"early": 0, "mid": 0, "late": 0}
    
    print("\nProcessing test scenarios:")
    print("-" * 70)
    
    for turns, winner, output in test_scenarios:
        death_reason, game_phase = runner._categorize_death(turns, winner, output)
        
        if death_reason:
            death_reasons[death_reason] = death_reasons.get(death_reason, 0) + 1
        
        if game_phase:
            death_by_phase[game_phase] = death_by_phase.get(game_phase, 0) + 1
        
        print(f"Turn {turns:3d} ({game_phase:5s}): {death_reason}")
    
    # Create the result structure as would be saved
    result_data = {
        "death_analysis": {
            "by_reason": death_reasons,
            "by_phase": death_by_phase
        }
    }
    
    print("\n" + "="*70)
    print("Expected Output Format:")
    print("="*70)
    print(json.dumps(result_data, indent=2))
    
    # Validation checks
    print("\n" + "="*70)
    print("Validation Checks:")
    print("="*70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: by_reason should contain actual death causes
    total_checks += 1
    actual_reasons = ["head_collision", "starvation", "self_collision", "body_collision", "wall_collision"]
    has_actual_reasons = any(reason in death_reasons for reason in actual_reasons)
    if has_actual_reasons:
        print("✓ by_reason contains actual death causes")
        checks_passed += 1
    else:
        print("✗ by_reason does NOT contain actual death causes")
    
    # Check 2: by_reason should NOT contain phase-based categories
    total_checks += 1
    phase_categories = ["early_game_death", "mid_game_death", "late_game_death"]
    has_phase_categories = any(cat in death_reasons for cat in phase_categories)
    if not has_phase_categories:
        print("✓ by_reason does NOT contain phase-based categories (CORRECT)")
        checks_passed += 1
    else:
        print("✗ by_reason contains phase-based categories (BUG NOT FIXED)")
    
    # Check 3: by_phase should only have early/mid/late keys
    total_checks += 1
    expected_phase_keys = {"early", "mid", "late"}
    phase_keys = set(death_by_phase.keys())
    if phase_keys == expected_phase_keys:
        print("✓ by_phase has correct keys (early, mid, late)")
        checks_passed += 1
    else:
        print(f"✗ by_phase has incorrect keys: {phase_keys}")
    
    # Check 4: Verify separation - no overlap between reasons and phases
    total_checks += 1
    overlap = set(death_reasons.keys()) & set(death_by_phase.keys())
    if not overlap:
        print("✓ No overlap between by_reason and by_phase (CORRECT)")
        checks_passed += 1
    else:
        print(f"✗ Overlap found between by_reason and by_phase: {overlap}")
    
    # Check 5: Verify specific death reasons are being tracked
    total_checks += 1
    expected_in_output = "head_collision"
    if expected_in_output in death_reasons:
        print(f"✓ '{expected_in_output}' found in by_reason")
        checks_passed += 1
    else:
        print(f"✗ '{expected_in_output}' NOT found in by_reason")
    
    print("\n" + "="*70)
    print(f"Test Results: {checks_passed}/{total_checks} checks passed")
    print("="*70)
    
    if checks_passed == total_checks:
        print("\n✓✓✓ ALL CHECKS PASSED - FIX IS WORKING CORRECTLY ✓✓✓\n")
        return 0
    else:
        print(f"\n✗✗✗ SOME CHECKS FAILED ({total_checks - checks_passed} failures) ✗✗✗\n")
        return 1


if __name__ == '__main__':
    exit_code = test_death_analysis_integration()
    sys.exit(exit_code)
