#!/usr/bin/env python3
"""
Test script to verify that all config.yaml parameters are accessible
and can be adjusted by the continuous training system.
"""

import yaml
import sys
from pathlib import Path

def test_config_coverage():
    """Test that all config parameters can be accessed and modified"""
    
    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ Error: config.yaml not found")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("Config Coverage Test")
    print("="*70)
    print()
    
    # Define all sections and their expected numeric parameters
    expected_sections = {
        'weights': ['space', 'head_collision', 'center_control', 'wall_penalty', 'cutoff', 'food'],
        'pursuit': ['distance_2', 'distance_3', 'distance_4', 'distance_5'],
        'traps': ['moderate', 'severe', 'critical', 'food_trap', 'food_trap_threshold'],
        'food_urgency': ['critical', 'low', 'normal'],
        'trapping': ['weight', 'space_cutoff_threshold', 'trapped_ratio'],
        'late_game': ['caution_multiplier', 'turn_threshold'],
        'hybrid': ['critical_health', 'critical_nearby_enemies', 'critical_space_ratio', 
                   'lookahead_depth', 'mcts_iterations', 'mcts_timeout_ms'],
        'search': ['max_astar_nodes', 'max_depth'],
        'optimization': ['learning_rate', 'discount_factor', 'exploration_rate', 
                        'batch_size', 'episodes']
    }
    
    total_params = 0
    accessible_params = 0
    issues = []
    
    for section, params in expected_sections.items():
        print(f"Section: {section}")
        
        if section not in config:
            issues.append(f"  ❌ Missing section: {section}")
            print(f"  ❌ Section not found in config")
            continue
        
        section_data = config[section]
        for param in params:
            total_params += 1
            
            if param not in section_data:
                issues.append(f"  ❌ Missing {section}.{param}")
                print(f"  ❌ {param}: NOT FOUND")
            else:
                value = section_data[param]
                if isinstance(value, (int, float)):
                    accessible_params += 1
                    print(f"  ✓ {param}: {value}")
                else:
                    issues.append(f"  ⚠ {section}.{param} is not numeric: {type(value)}")
                    print(f"  ⚠ {param}: {value} (type: {type(value).__name__})")
        
        print()
    
    # Summary
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Total expected parameters: {total_params}")
    print(f"Accessible numeric parameters: {accessible_params}")
    print(f"Coverage: {accessible_params}/{total_params} ({accessible_params/total_params*100:.1f}%)")
    print()
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
        print()
    
    if accessible_params == total_params:
        print("✅ SUCCESS: All parameters are accessible!")
        return True
    else:
        print(f"⚠ WARNING: {total_params - accessible_params} parameters not accessible")
        return False

if __name__ == '__main__':
    success = test_config_coverage()
    sys.exit(0 if success else 1)
