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
        """Run a single game and return the winner, turns, death reason, and game phase
        
        Returns:
            tuple: (winner, turns, death_reason, game_phase) where:
                - winner: str - Name of winning snake or "draw"/"error"
                - turns: int - Number of turns played
                - death_reason: str or None - Actual death reason (head_collision, starvation, etc.)
                - game_phase: str or None - Game phase when death occurred (early, mid, late)
        """
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
                
                # Get death reason and game phase separately
                death_reason, game_phase = self._categorize_death(turns, winner, output)
                return winner, turns, death_reason, game_phase
            
            # Check if both snakes died
            if "Game completed" in output and "no winner" in output.lower():
                phase = self._get_game_phase(0)
                return "draw", 0, "both_eliminated", phase
            
            return "error", 0, "parse_failed", None
            
        except subprocess.TimeoutExpired:
            return "error", 0, "timeout", None
        except Exception as e:
            return "error", 0, str(e), None
    
    def _get_game_phase(self, turns):
        """Determine game phase based on turn count"""
        if turns < 50:
            return "early"
        elif turns < 150:
            return "mid"
        else:
            return "late"
    
    def _categorize_death(self, turns, winner, output):
        """Categorize death reason and game phase separately
        
        Returns:
            tuple: (death_reason, game_phase) where:
                - death_reason: actual cause of death (head_collision, starvation, etc.) or None if won
                - game_phase: early/mid/late or None if won
        """
        # If we won, no death reason or phase needed
        if winner == "go-battleclank":
            return None, None
        
        # Determine game phase first
        game_phase = self._get_game_phase(turns)
        
        # Parse actual death reason from output (single pass optimization)
        output_lower = output.lower()
        death_reason = "unknown"  # Default if we can't determine specific reason
        
        if "eliminated" in output_lower:
            # Check for specific death causes
            if 'head' in output_lower and 'collision' in output_lower:
                death_reason = "head_collision"
            elif 'starvation' in output_lower or 'starved' in output_lower:
                death_reason = "starvation"
            elif 'self' in output_lower and 'collision' in output_lower:
                death_reason = "self_collision"
            elif 'body' in output_lower and 'collision' in output_lower:
                death_reason = "body_collision"
            elif 'wall' in output_lower and 'collision' in output_lower:
                death_reason = "wall_collision"
            elif 'collision' in output_lower:
                death_reason = "collision"
            elif 'trap' in output_lower:
                death_reason = "trapped"
            elif 'out of bounds' in output_lower:
                death_reason = "out_of_bounds"
        
        return death_reason, game_phase
    
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
        
        # Track death reasons and phases separately for failed games
        death_reasons = {}  # Actual causes: head_collision, starvation, etc.
        death_by_phase = {"early": 0, "mid": 0, "late": 0}  # Game phase when death occurred
        
        print("\n" + "="*60)
        print("  Running Games")
        print("="*60 + "\n")
        
        for i in range(1, self.num_games + 1):
            progress = i / self.num_games * 100
            print(f"Game {i}/{self.num_games} ({progress:.1f}%)...", end=" ", flush=True)
            
            winner, turns, death_reason, game_phase = self.run_game(i)
            
            if winner == "go-battleclank":
                wins += 1
                print(f"WIN (turns: {turns})")
            elif winner == "rust-baseline":
                losses += 1
                print(f"LOSS (turns: {turns})")
                
                # Track actual death reason
                if death_reason:
                    death_reasons[death_reason] = death_reasons.get(death_reason, 0) + 1
                    
                # Track by game phase
                if game_phase:
                    death_by_phase[game_phase] = death_by_phase.get(game_phase, 0) + 1
            elif winner == "draw":
                draws += 1
                print(f"DRAW")
            else:
                errors += 1
                error_msg = death_reason if death_reason else "unknown"
                print(f"ERROR ({error_msg})")
        
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
        
        # Print death reason analysis (actual causes)
        if death_reasons:
            print("Death Reason Analysis (by actual cause):")
            for reason, count in sorted(death_reasons.items(), key=lambda x: x[1], reverse=True):
                pct = (count / losses * 100) if losses > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}% of losses)")
            print()
        
        # Print death by game phase analysis
        total_phase_deaths = sum(death_by_phase.values())
        if total_phase_deaths > 0:
            print("Death by Game Phase:")
            for phase in ["early", "mid", "late"]:
                count = death_by_phase.get(phase, 0)
                if count > 0:
                    pct = (count / total_phase_deaths * 100) if total_phase_deaths > 0 else 0
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
                "by_phase": death_by_phase
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
