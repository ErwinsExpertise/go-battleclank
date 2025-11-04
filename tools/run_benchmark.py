#!/usr/bin/env python3
"""
Live benchmark runner for go-battleclank vs rust-baseline
Uses battlesnake CLI to run real games and parses the output
"""

import subprocess
import tempfile
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
            
            # Use temporary file for game state output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                game_state_file = f.name
            
            try:
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
                        '-r', str(game_num * 12345),  # Use game num as seed
                        '--output', game_state_file  # Save game state for death reason analysis
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
                    
                    # Get death reason from game state file
                    death_reason, game_phase = self._analyze_death_from_game_state(
                        game_state_file, turns, winner
                    )
                    return winner, turns, death_reason, game_phase
                
                # Check if both snakes died (draw)
                if "Game completed" in output and ("draw" in output.lower() or "no winner" in output.lower()):
                    # Parse turn count from draw message
                    draw_match = re.search(r'after (\d+) turns', output)
                    turns = int(draw_match.group(1)) if draw_match else 0
                    phase = self._get_game_phase(turns)
                    return "draw", turns, "both_eliminated", phase
                
                return "error", 0, "parse_failed", None
            finally:
                # Clean up temporary file
                try:
                    os.unlink(game_state_file)
                except (OSError, FileNotFoundError):
                    pass
            
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
    
    def _analyze_death_from_game_state(self, game_state_file, turns, winner):
        """Analyze game state JSON to determine death reason
        
        Args:
            game_state_file: Path to JSON file containing game states
            turns: Number of turns the game lasted
            winner: Name of the winning snake
            
        Returns:
            tuple: (death_reason, game_phase) or (None, None) if we won
        """
        # If we won, no death reason needed
        if winner == "go-battleclank":
            return None, None
        
        # Determine game phase
        game_phase = self._get_game_phase(turns)
        
        # Try to parse the game state file to determine death cause
        try:
            with open(game_state_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return "unknown", game_phase
            
            # Parse the last few turns to analyze what happened
            # The file contains one JSON object per line
            last_states = []
            for line in reversed(lines):
                try:
                    state = json.loads(line.strip())
                    # Look for turn data (has 'board' key)
                    if 'board' in state and 'snakes' in state['board']:
                        last_states.insert(0, state)
                        if len(last_states) >= 3:  # Get last 3 states
                            break
                except json.JSONDecodeError:
                    continue
            
            if not last_states:
                return "unknown", game_phase
            
            # Find our snake in the game state
            our_snake_id = None
            our_snake_name = "go-battleclank"
            
            # Look for our snake in the first state
            for snake in last_states[0]['board']['snakes']:
                if snake['name'] == our_snake_name:
                    our_snake_id = snake['id']
                    break
            
            if not our_snake_id:
                return "unknown", game_phase
            
            # Analyze death reason from state transitions
            death_reason = self._infer_death_reason(last_states, our_snake_id, our_snake_name)
            return death_reason, game_phase
            
        except Exception as e:
            # If we can't parse the game state, fall back to unknown
            return "unknown", game_phase
    
    def _infer_death_reason(self, states, our_snake_id, our_snake_name):
        """Infer death reason from game state transitions
        
        Args:
            states: List of game state dictionaries (last few turns)
            our_snake_id: ID of our snake
            our_snake_name: Name of our snake
            
        Returns:
            str: Death reason (starvation, head_collision, body_collision, wall_collision, self_collision, etc.)
        """
        if not states or len(states) < 1:
            return "unknown"
        
        # Find our snake in the last state where it existed
        our_snake = None
        last_state_with_snake = None
        state_before_death = None
        
        # Reverse states once for efficient iteration
        reversed_states = list(reversed(states))
        
        for i, state in enumerate(reversed_states):
            snakes = state['board']['snakes']
            for snake in snakes:
                if snake['id'] == our_snake_id:
                    our_snake = snake
                    last_state_with_snake = state
                    # Get the state before this one (if available)
                    if i + 1 < len(reversed_states):
                        state_before_death = reversed_states[i + 1]
                    break
            if our_snake:
                break
        
        if not our_snake:
            return "unknown"
        
        # Check for starvation (health reached 0)
        if our_snake['health'] <= 0:
            return "starvation"
        
        # Get board dimensions
        board_width = last_state_with_snake['board']['width']
        board_height = last_state_with_snake['board']['height']
        
        # Infer collision type by analyzing state transitions
        # Note: When a snake dies, it's removed from the board. The last state where
        # our snake exists (last_state_with_snake) is from the turn BEFORE death.
        # We analyze what happened on the next move to determine collision type.
        if state_before_death:
            # Find snake's position in previous state
            prev_snake = None
            for snake in state_before_death['board']['snakes']:
                if snake['id'] == our_snake_id:
                    prev_snake = snake
                    break
            
            if prev_snake:
                prev_head = prev_snake['head']
                curr_head = our_snake['head']
                
                # Calculate the move direction
                dx = curr_head['x'] - prev_head['x']
                dy = curr_head['y'] - prev_head['y']
                
                # Predict next position (where collision would have occurred)
                next_x = curr_head['x'] + dx
                next_y = curr_head['y'] + dy
                
                # Check for wall collision
                if next_x < 0 or next_x >= board_width or next_y < 0 or next_y >= board_height:
                    return "wall_collision"
                
                # Check for self collision at next position
                for segment in our_snake['body']:
                    if segment['x'] == next_x and segment['y'] == next_y:
                        return "self_collision"
                
                # Check for collision with other snakes at next position
                other_snakes = [s for s in last_state_with_snake['board']['snakes'] 
                               if s['id'] != our_snake_id]
                
                for other_snake in other_snakes:
                    # Check head-to-head collision
                    # Both snakes moved their head to same position
                    if (other_snake['head']['x'] == next_x and 
                        other_snake['head']['y'] == next_y):
                        # In head-to-head, smaller or equal length snake loses
                        if len(our_snake['body']) <= len(other_snake['body']):
                            return "head_collision"
                    
                    # Check body collision with current positions
                    for segment in other_snake['body']:
                        if segment['x'] == next_x and segment['y'] == next_y:
                            return "body_collision"
        
        # Fallback: analyze current state if we don't have previous state
        head = our_snake['head']
        
        # Check for self collision (head overlaps with body)
        if len(our_snake['body']) > 1:
            for segment in our_snake['body'][1:]:  # Skip head itself
                if segment['x'] == head['x'] and segment['y'] == head['y']:
                    return "self_collision"
        
        # Check for collision with other snakes in current state
        other_snakes = [s for s in last_state_with_snake['board']['snakes'] 
                       if s['id'] != our_snake_id]
        
        for other_snake in other_snakes:
            # Check head-to-head collision (heads at same position)
            if (other_snake['head']['x'] == head['x'] and 
                other_snake['head']['y'] == head['y']):
                if len(our_snake['body']) <= len(other_snake['body']):
                    return "head_collision"
            
            # Check body collision
            for segment in other_snake['body']:
                if segment['x'] == head['x'] and segment['y'] == head['y']:
                    return "body_collision"
        
        # Check if at board edge (possible wall collision)
        if (head['x'] == 0 or head['x'] == board_width - 1 or
            head['y'] == 0 or head['y'] == board_height - 1):
            return "wall_collision"
        
        # If we can't determine specific reason, return generic collision
        return "collision"
    
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
