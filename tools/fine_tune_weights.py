#!/usr/bin/env python3
"""
Fine-tune weights around the best baseline configuration
"""

import subprocess
import json
import os
import sys
import re
from datetime import datetime

class FineTuner:
    def __init__(self, games_per_test=50):
        self.games_per_test = games_per_test
        self.results = []
        self.best_win_rate = 0
        self.best_config = None
        
    def test_configuration(self, config_name, **kwargs):
        """Test a specific weight configuration"""
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"{'='*60}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print()
        
        # Update weights
        self.update_weights(**kwargs)
        
        # Rebuild
        print("Rebuilding...")
        result = subprocess.run(['go', 'build', '-o', 'go-battleclank'],
                              cwd='/home/runner/work/go-battleclank/go-battleclank',
                              capture_output=True)
        if result.returncode != 0:
            print(f"Build failed!")
            return None
        
        # Run benchmark
        print(f"Running {self.games_per_test} games...")
        result = subprocess.run(['python3', 'tools/run_benchmark.py', str(self.games_per_test)],
                              cwd='/home/runner/work/go-battleclank/go-battleclank',
                              capture_output=True, text=True, timeout=600)
        
        # Parse results
        output = result.stdout + result.stderr
        win_rate = self.parse_win_rate(output)
        losses, wins = self.parse_record(output)
        
        if win_rate is not None:
            result_data = {
                'config_name': config_name,
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'games': self.games_per_test,
                **kwargs
            }
            self.results.append(result_data)
            
            print(f"✓ Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
            
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.best_config = result_data
                print(f"🎯 NEW BEST! {win_rate:.1f}%")
            
            if win_rate >= 80.0:
                print(f"🎉 TARGET ACHIEVED!")
                return result_data
                
        return None
    
    def parse_win_rate(self, output):
        """Parse win rate from benchmark output"""
        for line in output.split('\n'):
            if 'Wins:' in line and '(' in line:
                try:
                    percent_str = line.split('(')[1].split('%')[0]
                    return float(percent_str)
                except:
                    pass
        return None
    
    def parse_record(self, output):
        """Parse wins and losses"""
        wins, losses = 0, 0
        for line in output.split('\n'):
            if 'Wins:' in line:
                try:
                    wins = int(line.split()[1])
                except:
                    pass
            if 'Losses:' in line:
                try:
                    losses = int(line.split()[1])
                except:
                    pass
        return losses, wins
    
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
\treturn &GreedySearch{{
\t\tSpaceWeight:         {space_weight},
\t\tHeadCollisionWeight: {head_collision_weight},
\t\tCenterWeight:        {center_weight},
\t\tWallPenaltyWeight:   {wall_penalty_weight},
\t\tCutoffWeight:        {cutoff_weight},
\t\tMaxDepth:            {max_depth},
\t\tUseAStar:            true,
\t\tMaxAStarNodes:       400,
\t}}
}}'''
        
        # Find and replace
        pattern = r'// NewGreedySearch.*?func NewGreedySearch\(\) \*GreedySearch \{[^}]*\n\}[^}]*\}'
        content = re.sub(pattern, new_func, content, flags=re.DOTALL)
        
        # Update multipliers
        content = re.sub(
            r'trapPenalty := heuristics\.GetSpaceTrapPenalty\(trapLevel\) \* [\d.]+',
            f'trapPenalty := heuristics.GetSpaceTrapPenalty(trapLevel) * {trap_multiplier}',
            content
        )
        
        content = re.sub(
            r'deadEndPenalty := heuristics\.EvaluateDeadEndAhead\(state, nextPos, g\.MaxDepth\) \* [\d.]+',
            f'deadEndPenalty := heuristics.EvaluateDeadEndAhead(state, nextPos, g.MaxDepth) * {dead_end_multiplier}',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def save_results(self):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fine_tune_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'games_per_test': self.games_per_test,
                'best_win_rate': self.best_win_rate,
                'best_config': self.best_config,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("FINE-TUNING SUMMARY")
        print("="*60)
        
        if self.best_config:
            print(f"\nBest Configuration: {self.best_config['config_name']}")
            print(f"Win Rate: {self.best_config['win_rate']:.1f}%")
            print(f"Record: {self.best_config['wins']}W / {self.best_config['losses']}L")
            print(f"\nWeights:")
            print(f"  SpaceWeight: {self.best_config['space_weight']}")
            print(f"  HeadCollisionWeight: {self.best_config['head_collision_weight']}")
            print(f"  CenterWeight: {self.best_config['center_weight']}")
            print(f"  WallPenaltyWeight: {self.best_config['wall_penalty_weight']}")
            print(f"  CutoffWeight: {self.best_config['cutoff_weight']}")
            print(f"  MaxDepth: {self.best_config['max_depth']}")
            print(f"  TrapMultiplier: {self.best_config['trap_multiplier']}")
            print(f"  DeadEndMultiplier: {self.best_config['dead_end_multiplier']}")

def main():
    tuner = FineTuner(games_per_test=50)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        Fine-Tuning Weights for 80% Win Rate               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\nGames per test: {tuner.games_per_test}")
    print("\nFine-tuning around baseline (best performer at 40%)...")
    
    # Baseline reference: space=250, head=600, center=10, wall=150, cutoff=350, depth=35, trap=0.5, dead=0.5
    
    # Test variations around baseline
    configs = [
        # Baseline (verify)
        ("Baseline", 250, 600, 10, 150, 350, 35, 0.5, 0.5),
        
        # Reduce trap/dead end penalties significantly (may be too conservative)
        ("Low Penalties", 250, 600, 10, 150, 350, 35, 0.2, 0.2),
        
        # Moderate penalties
        ("Moderate Penalties", 250, 600, 10, 150, 350, 35, 0.35, 0.35),
        
        # Higher space weight
        ("High Space", 320, 600, 10, 150, 350, 35, 0.4, 0.4),
        
        # Lower head collision penalty (may be too high)
        ("Lower Head Collision", 250, 500, 10, 150, 350, 35, 0.4, 0.4),
        
        # Combination: high space, low penalties
        ("Space + Low Penalties", 320, 600, 10, 150, 350, 35, 0.3, 0.3),
        
        # Deeper search with moderate penalties
        ("Deep + Moderate", 250, 600, 10, 150, 350, 40, 0.4, 0.4),
        
        # Very low penalties (most aggressive)
        ("Minimal Penalties", 250, 600, 10, 150, 350, 35, 0.15, 0.15),
    ]
    
    for config in configs:
        result = tuner.test_configuration(
            config[0],
            space_weight=config[1],
            head_collision_weight=config[2],
            center_weight=config[3],
            wall_penalty_weight=config[4],
            cutoff_weight=config[5],
            max_depth=config[6],
            trap_multiplier=config[7],
            dead_end_multiplier=config[8]
        )
        
        if result and result['win_rate'] >= 80.0:
            print(f"\n✓ SUCCESS: Found configuration achieving 80%+!")
            break
    
    tuner.save_results()
    
    if tuner.best_win_rate >= 80.0:
        return 0
    else:
        print(f"\n⚠ Best: {tuner.best_win_rate:.1f}% (target: 80%)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
