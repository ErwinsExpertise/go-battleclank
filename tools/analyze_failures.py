#!/usr/bin/env python3
"""
Analyze game failures to identify patterns and suggest improvements
Runs games with detailed logging and analyzes death scenarios
"""

import subprocess
import json
import re
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime

class FailureAnalyzer:
    def __init__(self, num_games=50):
        self.num_games = num_games
        self.failures = []
        self.go_port = 8000
        self.rust_port = 8080
        
    def run_analysis(self):
        """Run games and analyze failures"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║         Failure Pattern Analysis                          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"\nRunning {self.num_games} games with detailed analysis...")
        
        # Start servers
        self.start_servers()
        
        # Run games
        wins = 0
        losses = 0
        
        for i in range(1, self.num_games + 1):
            print(f"\nGame {i}/{self.num_games}...", end=" ", flush=True)
            
            winner, turns, details = self.run_single_game(i)
            
            if winner == "go-battleclank":
                wins += 1
                print(f"WIN (turns: {turns})")
            elif winner == "rust-baseline":
                losses += 1
                print(f"LOSS (turns: {turns})")
                self.failures.append({
                    'game': i,
                    'turns': turns,
                    'details': details
                })
            else:
                print(f"ERROR")
        
        self.stop_servers()
        
        # Analyze failures
        self.analyze_patterns()
        
        # Generate report
        self.generate_report(wins, losses)
        
    def run_single_game(self, game_num):
        """Run a single game and capture details"""
        try:
            gopath = subprocess.check_output(['go', 'env', 'GOPATH'], text=True).strip()
            battlesnake_path = f"{gopath}/bin/battlesnake"
            
            result = subprocess.run(
                [
                    battlesnake_path, 'play',
                    '-W', '11',
                    '-H', '11',
                    '-t', '500',
                    '--name', 'go-battleclank',
                    '--url', f'http://localhost:{self.go_port}',
                    '--name', 'rust-baseline',
                    '--url', f'http://localhost:{self.rust_port}',
                    '--sequential',
                    '-r', str(game_num * 12345)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Parse winner and turns
            match = re.search(r'Game completed after (\d+) turns\. (.+?) was the winner\.', output)
            if match:
                turns = int(match.group(1))
                winner = match.group(2)
                
                # Extract details about the game
                details = self.extract_game_details(output, turns)
                
                return winner, turns, details
            
            return "error", 0, {}
            
        except Exception as e:
            return "error", 0, {'error': str(e)}
    
    def extract_game_details(self, output, final_turns):
        """Extract details about the game for analysis"""
        details = {
            'final_turns': final_turns,
            'turn_category': self.categorize_turn_count(final_turns)
        }
        
        # Categorize early, mid, late game deaths
        return details
    
    def categorize_turn_count(self, turns):
        """Categorize when the snake died"""
        if turns < 50:
            return "early_game"
        elif turns < 150:
            return "mid_game"
        elif turns < 300:
            return "late_game"
        else:
            return "very_late_game"
    
    def analyze_patterns(self):
        """Analyze failure patterns"""
        print("\n" + "="*60)
        print("FAILURE PATTERN ANALYSIS")
        print("="*60)
        
        if not self.failures:
            print("\nNo failures to analyze!")
            return
        
        # Turn distribution
        turn_categories = Counter([f['details']['turn_category'] for f in self.failures])
        
        print(f"\nTotal Failures: {len(self.failures)}")
        print(f"\nFailure Timing Distribution:")
        for category, count in turn_categories.most_common():
            percentage = (count / len(self.failures)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Turn statistics
        turns_list = [f['turns'] for f in self.failures]
        avg_turns = sum(turns_list) / len(turns_list)
        min_turns = min(turns_list)
        max_turns = max(turns_list)
        
        print(f"\nTurn Statistics:")
        print(f"  Average: {avg_turns:.1f}")
        print(f"  Min: {min_turns}")
        print(f"  Max: {max_turns}")
        
        # Identify patterns
        early_deaths = [f for f in self.failures if f['details']['turn_category'] == 'early_game']
        if early_deaths:
            print(f"\n⚠ High early game deaths: {len(early_deaths)} games")
            print("  Suggestion: May be too aggressive early or poor opening strategy")
        
        mid_deaths = [f for f in self.failures if f['details']['turn_category'] == 'mid_game']
        if mid_deaths:
            print(f"\n⚠ Mid game deaths: {len(mid_deaths)} games")
            print("  Suggestion: Food control or trap avoidance issues")
        
        late_deaths = [f for f in self.failures if f['details']['turn_category'] in ['late_game', 'very_late_game']]
        if late_deaths:
            print(f"\n✓ Late game competitive: {len(late_deaths)} games")
            print("  Suggestion: Close matches - small improvements could help")
    
    def generate_report(self, wins, losses):
        """Generate detailed report"""
        win_rate = (wins / self.num_games) * 100 if self.num_games > 0 else 0
        
        print("\n" + "="*60)
        print("OVERALL RESULTS")
        print("="*60)
        print(f"\nWins: {wins} ({win_rate:.1f}%)")
        print(f"Losses: {losses} ({(losses/self.num_games)*100:.1f}%)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failure_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'num_games': self.num_games,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'failures': self.failures
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if win_rate < 40:
            print("\n1. Try increasing food seeking aggressiveness")
            print("2. Reduce trap detection penalties")
            print("3. Test MCTS for better tactical planning")
        elif win_rate < 60:
            print("\n1. Fine-tune A-Star parameters (MaxAStarNodes)")
            print("2. Improve early game strategy")
            print("3. Better food control in contested situations")
        else:
            print("\n1. Focus on edge cases causing remaining losses")
            print("2. Consider opponent-specific adaptations")
            print("3. Optimize endgame scenarios")
    
    def start_servers(self):
        """Start both snake servers"""
        print("\nStarting snake servers...")
        
        # Start Go snake
        env = os.environ.copy()
        env['PORT'] = str(self.go_port)
        self.go_process = subprocess.Popen(
            ['./go-battleclank'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd='/home/runner/work/go-battleclank/go-battleclank'
        )
        
        # Start Rust baseline
        env = os.environ.copy()
        env['BIND_PORT'] = str(self.rust_port)
        self.rust_process = subprocess.Popen(
            ['./baseline/target/release/baseline-snake'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd='/home/runner/work/go-battleclank/go-battleclank'
        )
        
        import time
        time.sleep(3)
        print("Servers started")
    
    def stop_servers(self):
        """Stop both servers"""
        print("\nStopping servers...")
        if hasattr(self, 'go_process'):
            self.go_process.terminate()
        if hasattr(self, 'rust_process'):
            self.rust_process.terminate()

def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    analyzer = FailureAnalyzer(num_games=num_games)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
