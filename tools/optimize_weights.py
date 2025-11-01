#!/usr/bin/env python3
"""
Weight optimization tool for achieving 80% win rate vs baseline
Tests different weight configurations and finds optimal balance
"""

import subprocess
import json
import os
import sys
from datetime import datetime
import time

class WeightOptimizer:
    def __init__(self, games_per_test=30, target_win_rate=80.0):
        self.games_per_test = games_per_test
        self.target_win_rate = target_win_rate
        self.results = []
        
    def test_configuration(self, config_name, space_weight, head_collision_weight, 
                          center_weight, wall_penalty_weight, cutoff_weight, 
                          max_depth, trap_multiplier, dead_end_multiplier):
        """Test a specific weight configuration"""
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {config_name}")
        print(f"{'='*60}")
        print(f"SpaceWeight: {space_weight}")
        print(f"HeadCollisionWeight: {head_collision_weight}")
        print(f"CenterWeight: {center_weight}")
        print(f"WallPenaltyWeight: {wall_penalty_weight}")
        print(f"CutoffWeight: {cutoff_weight}")
        print(f"MaxDepth: {max_depth}")
        print(f"TrapMultiplier: {trap_multiplier}")
        print(f"DeadEndMultiplier: {dead_end_multiplier}")
        print()
        
        # Update the greedy.go file with new weights
        self.update_weights(space_weight, head_collision_weight, center_weight,
                           wall_penalty_weight, cutoff_weight, max_depth,
                           trap_multiplier, dead_end_multiplier)
        
        # Rebuild the snake
        print("Rebuilding snake...")
        result = subprocess.run(['go', 'build', '-o', 'go-battleclank'],
                              cwd='/home/runner/work/go-battleclank/go-battleclank',
                              capture_output=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr.decode()}")
            return None
        
        # Run benchmark
        print(f"Running {self.games_per_test} games...")
        result = subprocess.run(['python3', 'tools/run_benchmark.py', str(self.games_per_test)],
                              cwd='/home/runner/work/go-battleclank/go-battleclank',
                              capture_output=True, text=True)
        
        # Parse results
        output = result.stdout + result.stderr
        win_rate = self.parse_win_rate(output)
        
        if win_rate is not None:
            result_data = {
                'config_name': config_name,
                'win_rate': win_rate,
                'space_weight': space_weight,
                'head_collision_weight': head_collision_weight,
                'center_weight': center_weight,
                'wall_penalty_weight': wall_penalty_weight,
                'cutoff_weight': cutoff_weight,
                'max_depth': max_depth,
                'trap_multiplier': trap_multiplier,
                'dead_end_multiplier': dead_end_multiplier,
                'games': self.games_per_test
            }
            self.results.append(result_data)
            
            print(f"✓ Win Rate: {win_rate:.1f}%")
            
            if win_rate >= self.target_win_rate:
                print(f"🎉 TARGET ACHIEVED! Win rate {win_rate:.1f}% >= {self.target_win_rate}%")
                return result_data
        else:
            print("✗ Failed to parse results")
            
        return None
    
    def parse_win_rate(self, output):
        """Parse win rate from benchmark output"""
        for line in output.split('\n'):
            if 'Wins:' in line and '(' in line:
                # Extract percentage from line like "Wins:   15 (50.0%)"
                try:
                    percent_str = line.split('(')[1].split('%')[0]
                    return float(percent_str)
                except:
                    pass
        return None
    
    def update_weights(self, space_weight, head_collision_weight, center_weight,
                      wall_penalty_weight, cutoff_weight, max_depth,
                      trap_multiplier, dead_end_multiplier):
        """Update weights in greedy.go"""
        file_path = '/home/runner/work/go-battleclank/go-battleclank/algorithms/search/greedy.go'
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace NewGreedySearch function
        new_func = f'''// NewGreedySearch creates a new greedy search with tuned weights
// AUTO-TUNED for baseline opponent
func NewGreedySearch() *GreedySearch {{
	return &GreedySearch{{
		SpaceWeight:         {space_weight},
		HeadCollisionWeight: {head_collision_weight},
		CenterWeight:        {center_weight},
		WallPenaltyWeight:   {wall_penalty_weight},
		CutoffWeight:        {cutoff_weight},
		MaxDepth:            {max_depth},
		UseAStar:            true,
		MaxAStarNodes:       400,
	}}
}}'''
        
        # Find and replace the NewGreedySearch function
        import re
        pattern = r'// NewGreedySearch.*?func NewGreedySearch\(\) \*GreedySearch \{[^}]*\n\}[^}]*\}'
        content = re.sub(pattern, new_func, content, flags=re.DOTALL)
        
        # Update trap penalty multiplier
        content = re.sub(
            r'trapPenalty := heuristics\.GetSpaceTrapPenalty\(trapLevel\) \* [\d.]+',
            f'trapPenalty := heuristics.GetSpaceTrapPenalty(trapLevel) * {trap_multiplier}',
            content
        )
        
        # Update dead end multiplier
        content = re.sub(
            r'deadEndPenalty := heuristics\.EvaluateDeadEndAhead\(state, nextPos, g\.MaxDepth\) \* [\d.]+',
            f'deadEndPenalty := heuristics.EvaluateDeadEndAhead(state, nextPos, g.MaxDepth) * {dead_end_multiplier}',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def save_results(self):
        """Save optimization results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'target_win_rate': self.target_win_rate,
                'games_per_test': self.games_per_test,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.results:
            # Sort by win rate
            sorted_results = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)
            
            print(f"\nBest Configuration: {sorted_results[0]['config_name']}")
            print(f"Win Rate: {sorted_results[0]['win_rate']:.1f}%")
            print(f"SpaceWeight: {sorted_results[0]['space_weight']}")
            print(f"HeadCollisionWeight: {sorted_results[0]['head_collision_weight']}")
            print(f"CenterWeight: {sorted_results[0]['center_weight']}")
            print(f"WallPenaltyWeight: {sorted_results[0]['wall_penalty_weight']}")
            print(f"CutoffWeight: {sorted_results[0]['cutoff_weight']}")
            print(f"MaxDepth: {sorted_results[0]['max_depth']}")
            print(f"TrapMultiplier: {sorted_results[0]['trap_multiplier']}")
            print(f"DeadEndMultiplier: {sorted_results[0]['dead_end_multiplier']}")
            
            print("\n" + "="*60)
            print("All Results (sorted by win rate):")
            print("="*60)
            for i, result in enumerate(sorted_results, 1):
                status = "✓ TARGET" if result['win_rate'] >= self.target_win_rate else "✗ BELOW"
                print(f"{i}. {result['config_name']}: {result['win_rate']:.1f}% {status}")

def main():
    optimizer = WeightOptimizer(games_per_test=30, target_win_rate=80.0)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Weight Optimization for 80% Win Rate Target           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\nGames per test: {optimizer.games_per_test}")
    print(f"Target win rate: {optimizer.target_win_rate}%")
    print("\nStarting systematic weight optimization...")
    
    # Test configurations - systematic exploration
    configurations = [
        # Baseline (current)
        ("Baseline", 250, 600, 10, 150, 350, 35, 0.5, 0.5),
        
        # Less conservative (reduce trap penalties)
        ("Less Conservative", 250, 600, 10, 150, 350, 35, 0.3, 0.3),
        
        # More aggressive space seeking
        ("Aggressive Space", 350, 600, 15, 150, 350, 40, 0.4, 0.4),
        
        # Balanced approach
        ("Balanced", 300, 650, 12, 175, 375, 38, 0.4, 0.4),
        
        # Food focused (lower penalties, higher space)
        ("Food Focused", 320, 550, 12, 140, 340, 35, 0.35, 0.35),
        
        # Survival focused (higher penalties)
        ("Survival Focused", 280, 700, 10, 200, 400, 40, 0.6, 0.6),
        
        # Deep search
        ("Deep Search", 300, 650, 12, 175, 375, 45, 0.45, 0.45),
        
        # Ultra aggressive
        ("Ultra Aggressive", 380, 650, 18, 160, 380, 42, 0.35, 0.35),
    ]
    
    best_config = None
    
    for config in configurations:
        result = optimizer.test_configuration(*config)
        
        if result and result['win_rate'] >= optimizer.target_win_rate:
            best_config = result
            print(f"\n🎯 Target achieved with {config[0]}!")
            # Continue testing to see if we can do even better
    
    optimizer.save_results()
    
    if best_config:
        print(f"\n✓ SUCCESS: Found configuration achieving {best_config['win_rate']:.1f}% win rate")
        return 0
    else:
        print(f"\n⚠ No configuration reached {optimizer.target_win_rate}% target")
        print("   Running additional fine-tuning based on best result...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
