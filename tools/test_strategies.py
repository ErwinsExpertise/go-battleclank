#!/usr/bin/env python3
"""
Test different strategies (MCTS, Lookahead, A-Star variations)
to find improvements toward 80% win rate
"""

import subprocess
import json
import re
import os
import sys
from datetime import datetime

class StrategyTester:
    def __init__(self, games_per_test=50):
        self.games_per_test = games_per_test
        self.results = []
        
    def test_strategy(self, name, strategy_type, params=None):
        """Test a specific strategy"""
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {name}")
        print(f"Type: {strategy_type}")
        if params:
            print(f"Parameters: {params}")
        print('='*60)
        
        # Update strategy in logic_refactored.go
        if strategy_type == "mcts":
            self.enable_mcts(params)
        elif strategy_type == "lookahead":
            self.enable_lookahead(params)
        elif strategy_type == "greedy-astar":
            self.tune_astar(params)
        else:
            print(f"Unknown strategy type: {strategy_type}")
            return None
        
        # Rebuild
        print("Rebuilding...")
        result = subprocess.run(
            ['go', 'build', '-o', 'go-battleclank'],
            cwd='/home/runner/work/go-battleclank/go-battleclank',
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr.decode()}")
            return None
        
        # Run benchmark
        print(f"Running {self.games_per_test} games...")
        result = subprocess.run(
            ['python3', 'tools/run_benchmark.py', str(self.games_per_test)],
            cwd='/home/runner/work/go-battleclank/go-battleclank',
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # Parse results
        output = result.stdout + result.stderr
        win_rate = self.parse_win_rate(output)
        wins, losses = self.parse_record(output)
        
        if win_rate is not None:
            result_data = {
                'name': name,
                'strategy_type': strategy_type,
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'params': params
            }
            self.results.append(result_data)
            
            print(f"✓ Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
            
            if win_rate >= 80.0:
                print(f"🎉 TARGET ACHIEVED!")
                return result_data
        else:
            print("✗ Failed to parse results")
        
        return None
    
    def enable_mcts(self, params):
        """Switch to MCTS strategy"""
        file_path = '/home/runner/work/go-battleclank/go-battleclank/logic_refactored.go'
        
        max_iterations = params.get('max_iterations', 100) if params else 100
        max_time_ms = params.get('max_time_ms', 400) if params else 400
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace strategy creation
        old_strategy = 'strategy := search.NewGreedySearch()'
        new_strategy = f'strategy := search.NewMCTSSearch({max_iterations}, {max_time_ms}*time.Millisecond)'
        
        content = content.replace(old_strategy, new_strategy)
        
        # Add time import if not present
        if 'import (\n\t"log"\n\t"time"' not in content:
            content = content.replace(
                'import (\n\t"log"',
                'import (\n\t"log"\n\t"time"'
            )
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def enable_lookahead(self, params):
        """Switch to Lookahead strategy"""
        file_path = '/home/runner/work/go-battleclank/go-battleclank/logic_refactored.go'
        
        max_depth = params.get('max_depth', 3) if params else 3
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace strategy creation
        old_strategy = 'strategy := search.NewGreedySearch()'
        new_strategy = f'strategy := search.NewLookaheadSearch({max_depth})'
        
        content = content.replace(old_strategy, new_strategy)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def tune_astar(self, params):
        """Tune A-Star parameters in greedy search"""
        file_path = '/home/runner/work/go-battleclank/go-battleclank/algorithms/search/greedy.go'
        
        max_astar_nodes = params.get('max_astar_nodes', 400) if params else 400
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace MaxAStarNodes value
        content = re.sub(
            r'MaxAStarNodes:\s+\d+,',
            f'MaxAStarNodes:       {max_astar_nodes},',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def parse_win_rate(self, output):
        """Parse win rate from output"""
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
        return wins, losses
    
    def restore_greedy(self):
        """Restore greedy strategy as default"""
        file_path = '/home/runner/work/go-battleclank/go-battleclank/logic_refactored.go'
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace any strategy with greedy
        content = re.sub(
            r'strategy := search\.New\w+Search\([^)]*\)',
            'strategy := search.NewGreedySearch()',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'games_per_test': self.games_per_test,
                'results': self.results
            }, f, indent=2)
        
        print(f"\n\nResults saved to: {filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("STRATEGY TEST SUMMARY")
        print("="*60)
        
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x['win_rate'], reverse=True)
            
            print(f"\nBest Strategy: {sorted_results[0]['name']}")
            print(f"Win Rate: {sorted_results[0]['win_rate']:.1f}%")
            print(f"Type: {sorted_results[0]['strategy_type']}")
            
            print("\n" + "="*60)
            print("All Results (sorted by win rate):")
            print("="*60)
            for i, result in enumerate(sorted_results, 1):
                status = "✓ TARGET" if result['win_rate'] >= 80.0 else "✗ BELOW"
                print(f"{i}. {result['name']}: {result['win_rate']:.1f}% {status}")

def main():
    tester = StrategyTester(games_per_test=50)
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      Strategy Testing for 80% Win Rate                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\nGames per strategy: {tester.games_per_test}")
    
    # Test strategies
    strategies = [
        # Baseline
        ("Baseline Greedy", "greedy-astar", {'max_astar_nodes': 400}),
        
        # A-Star variations
        ("Greedy + A* 200", "greedy-astar", {'max_astar_nodes': 200}),
        ("Greedy + A* 600", "greedy-astar", {'max_astar_nodes': 600}),
        ("Greedy + A* 800", "greedy-astar", {'max_astar_nodes': 800}),
        
        # MCTS variations
        ("MCTS 50 iter 400ms", "mcts", {'max_iterations': 50, 'max_time_ms': 400}),
        ("MCTS 100 iter 400ms", "mcts", {'max_iterations': 100, 'max_time_ms': 400}),
        ("MCTS 200 iter 450ms", "mcts", {'max_iterations': 200, 'max_time_ms': 450}),
        
        # Lookahead
        ("Lookahead depth 2", "lookahead", {'max_depth': 2}),
        ("Lookahead depth 3", "lookahead", {'max_depth': 3}),
    ]
    
    best_result = None
    
    for name, strategy_type, params in strategies:
        result = tester.test_strategy(name, strategy_type, params)
        
        if result and result['win_rate'] >= 80.0:
            best_result = result
            print(f"\n🎯 Target achieved with {name}!")
            break
        
        # Always restore greedy as baseline
        tester.restore_greedy()
    
    # Restore greedy at end
    tester.restore_greedy()
    
    tester.save_results()
    
    if best_result:
        print(f"\n✓ SUCCESS: {best_result['name']} achieved {best_result['win_rate']:.1f}%")
        return 0
    else:
        best = max(tester.results, key=lambda x: x['win_rate']) if tester.results else None
        if best:
            print(f"\n⚠ Best: {best['name']} at {best['win_rate']:.1f}% (target: 80%)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
