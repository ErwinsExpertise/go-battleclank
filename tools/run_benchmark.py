#!/usr/bin/env python3
"""
Live benchmark runner for go-battleclank vs rust-baseline
Uses battlesnake CLI to run real games and parses the output
"""

import subprocess
import time
import os
import sys
import signal
import json
import re
from datetime import datetime

class BenchmarkRunner:
    def __init__(self, num_games=100, board_size=11, max_turns=500, go_port=8000, rust_port=8080, manage_servers=True, config_file=None):
        self.num_games = num_games
        self.board_size = board_size
        self.max_turns = max_turns
        self.go_port = go_port
        self.rust_port = rust_port
        self.manage_servers = manage_servers  # Whether to start/stop servers
        self.config_file = config_file  # Optional config file path for the Go snake
        self.go_process = None
        self.rust_process = None
        
    def start_snakes(self):
        """Start both snake servers"""
        print("Starting snake servers...")
        
        # Start Go snake
        print(f"  Starting Go snake on port {self.go_port}...", end=" ", flush=True)
        env = os.environ.copy()
        env['PORT'] = str(self.go_port)
        self.go_process = subprocess.Popen(
            ['./go-battleclank'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )
        time.sleep(2)
        if self.go_process.poll() is not None:
            print("FAILED")
            sys.exit(1)
        print("OK")
        
        # Start Rust baseline
        print(f"  Starting Rust baseline on port {self.rust_port}...", end=" ", flush=True)
        env = os.environ.copy()
        env['BIND_PORT'] = str(self.rust_port)
        self.rust_process = subprocess.Popen(
            ['./baseline/target/release/baseline-snake'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )
        time.sleep(2)
        if self.rust_process.poll() is not None:
            print("FAILED")
            self.cleanup()
            sys.exit(1)
        print("OK")
        
        # Wait for servers to be ready
        print("  Waiting for servers to be ready...")
        time.sleep(2)
        
    def cleanup(self):
        """Stop both snake servers"""
        print("\nStopping snake servers...")
        if self.go_process:
            self.go_process.terminate()
            try:
                self.go_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.go_process.kill()
        if self.rust_process:
            self.rust_process.terminate()
            try:
                self.rust_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rust_process.kill()
    
    def run_game(self, game_num):
        """Run a single game and return the winner, turns, and death context"""
        try:
            # Get battlesnake path
            gopath = subprocess.check_output(['go', 'env', 'GOPATH'], text=True).strip()
            battlesnake_path = f"{gopath}/bin/battlesnake"
            
            result = subprocess.run(
                [
                    battlesnake_path, 'play',
                    '-W', str(self.board_size),
                    '-H', str(self.board_size),
                    '-t', str(self.max_turns),
                    '--name', 'go-battleclank',
                    '--url', f'http://localhost:{self.go_port}',
                    '--name', 'rust-baseline',
                    '--url', f'http://localhost:{self.rust_port}',
                    '--sequential',
                    '-r', str(game_num * 12345)  # Use game num as seed
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Parse winner from output
            # Look for line like: "Game completed after N turns. WINNER was the winner."
            match = re.search(r'Game completed after (\d+) turns\. (.+?) was the winner\.', output)
            if match:
                turns = int(match.group(1))
                winner = match.group(2)
                
                # Categorize death reason based on turn number and context
                death_context = self._categorize_death(turns, winner, output)
                return winner, turns, death_context
            
            # Check if both snakes died
            if "Game completed" in output and "no winner" in output.lower():
                return "draw", 0, "both-eliminated"
            
            return "error", 0, "parse-failed"
            
        except subprocess.TimeoutExpired:
            return "error", 0, "timeout"
        except Exception as e:
            return "error", 0, str(e)
    
    def _categorize_death(self, turns, winner, output):
        """Categorize death reason based on available context"""
        # If we lost, try to categorize why
        if winner != "go-battleclank":
            # Check for specific patterns in output
            if "eliminated" in output.lower():
                # Try to extract elimination reason from output
                for line in output.split('\n'):
                    if 'eliminated' in line.lower():
                        if 'collision' in line.lower():
                            return "collision"
                        elif 'starvation' in line.lower() or 'starved' in line.lower():
                            return "starvation"
                        elif 'head' in line.lower():
                            return "head_collision"
            
            # Categorize by game phase
            if turns < 50:
                return "early_game_death"
            elif turns < 150:
                return "mid_game_death"
            else:
                return "late_game_death"
        
        return None
    
    def run_benchmark(self):
        """Run all games and collect statistics"""
        print("="*60)
        print("  Live Battlesnake Benchmark")
        print("  Go Snake vs Rust Baseline")
        print("="*60)
        print(f"Games: {self.num_games}")
        print(f"Board: {self.board_size}x{self.board_size}")
        print(f"Max turns: {self.max_turns}")
        if self.config_file:
            print(f"Config file: {self.config_file}")
        print()
        
        if self.manage_servers:
            self.start_snakes()
        else:
            print("Using existing servers (not managing server lifecycle)")
        
        wins = 0
        losses = 0
        draws = 0
        errors = 0
        
        # Track death reasons for failed games
        death_reasons = {}
        turn_stats = {"early": 0, "mid": 0, "late": 0}
        
        print("\n" + "="*60)
        print("  Running Games")
        print("="*60 + "\n")
        
        for i in range(1, self.num_games + 1):
            progress = i / self.num_games * 100
            print(f"Game {i}/{self.num_games} ({progress:.1f}%)...", end=" ", flush=True)
            
            winner, turns, death_context = self.run_game(i)
            
            if winner == "go-battleclank":
                wins += 1
                print(f"WIN (turns: {turns})")
            elif winner == "rust-baseline":
                losses += 1
                print(f"LOSS (turns: {turns})")
                
                # Track death reason
                if death_context:
                    death_reasons[death_context] = death_reasons.get(death_context, 0) + 1
                    
                    # Track by game phase
                    if turns < 50:
                        turn_stats["early"] += 1
                    elif turns < 150:
                        turn_stats["mid"] += 1
                    else:
                        turn_stats["late"] += 1
            elif winner == "draw":
                draws += 1
                print(f"DRAW")
            else:
                errors += 1
                print(f"ERROR ({death_context})")
        
        if self.manage_servers:
            self.cleanup()
        
        # Print results
        win_rate = (wins / self.num_games) * 100 if self.num_games > 0 else 0
        
        print("\n" + "="*60)
        print("  Results Summary")
        print("="*60)
        print(f"Wins:   {wins} ({win_rate:.1f}%)")
        print(f"Losses: {losses} ({(losses/self.num_games)*100:.1f}%)")
        print(f"Draws:  {draws} ({(draws/self.num_games)*100:.1f}%)")
        print(f"Errors: {errors}\n")
        
        # Print death reason analysis
        if death_reasons:
            print("Death Reason Analysis:")
            for reason, count in sorted(death_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = (count / losses * 100) if losses > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}% of losses)")
            print()
            
            print("Death by Game Phase:")
            for phase, count in turn_stats.items():
                pct = (count / losses * 100) if losses > 0 else 0
                print(f"  {phase}: {count} ({pct:.1f}% of losses)")
            print()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "timestamp": timestamp,
            "config": {
                "num_games": self.num_games,
                "board_size": self.board_size,
                "max_turns": self.max_turns
            },
            "results": {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "errors": errors,
                "win_rate": win_rate
            },
            "death_analysis": {
                "by_reason": death_reasons,
                "by_phase": turn_stats
            }
        }
        
        os.makedirs("benchmark_results_live", exist_ok=True)
        with open(f"benchmark_results_live/results_{timestamp}.json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Assessment
        if win_rate >= 80.0:
            print("✓ SUCCESS: Win rate >= 80% target!")
            return 0
        elif win_rate >= 60.0:
            print("⚠ PARTIAL: Win rate >= 60%")
            return 0
        else:
            print("✗ BELOW TARGET: Win rate < 60%")
            return 1

def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    go_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    rust_port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
    manage_servers = sys.argv[4].lower() != 'no-manage' if len(sys.argv) > 4 else True
    config_file = sys.argv[5] if len(sys.argv) > 5 else None
    
    runner = BenchmarkRunner(
        num_games=num_games, 
        go_port=go_port, 
        rust_port=rust_port,
        manage_servers=manage_servers,
        config_file=config_file
    )
    
    # Setup signal handler for cleanup
    def signal_handler(sig, frame):
        print("\nInterrupted! Cleaning up...")
        runner.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        exit_code = runner.run_benchmark()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        runner.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
