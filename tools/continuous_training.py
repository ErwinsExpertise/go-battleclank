#!/usr/bin/env python3
"""
24/7 Continuous Neural Network Training for Battlesnake Weight Optimization
Runs indefinitely, automatically managing checkpoints and improvements
"""

import torch
import subprocess
import json
import yaml
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class ContinuousTrainer:
    """Manages 24/7 continuous training with automatic checkpointing and recovery"""
    
    def __init__(self, 
                 games_per_iteration=30,
                 checkpoint_interval=10,
                 max_iterations=None,
                 min_improvement=0.001):
        self.games_per_iteration = games_per_iteration
        self.checkpoint_interval = checkpoint_interval
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        
        self.results_dir = Path("nn_training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.checkpoint_file = self.results_dir / "checkpoint.json"
        self.best_config_file = Path("config.yaml")
        self.global_log_file = self.results_dir / "training_log.jsonl"
        
        # Load or initialize state
        self.state = self.load_checkpoint()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        self.running = True
        
    def load_checkpoint(self):
        """Load checkpoint or create new state"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                print(f"✓ Resumed from checkpoint: iteration {state['iteration']}, best win rate {state['best_win_rate']:.1%}")
                return state
        else:
            state = {
                'iteration': 0,
                'best_win_rate': 0.0,
                'best_config': None,
                'total_games': 0,
                'improvements': 0,
                'start_time': datetime.now().isoformat(),
                'last_improvement_iteration': 0
            }
            print("✓ Starting fresh training session")
            return state
    
    def save_checkpoint(self):
        """Save current state to checkpoint"""
        self.state['last_checkpoint'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def log_iteration(self, iteration_data):
        """Append iteration results to log file"""
        with open(self.global_log_file, 'a') as f:
            f.write(json.dumps(iteration_data) + '\n')
    
    def run_benchmark(self, num_games):
        """Run benchmark and return win rate"""
        print(f"  Running {num_games}-game benchmark...")
        try:
            result = subprocess.run(
                ['python3', 'tools/run_benchmark.py', str(num_games)],
                capture_output=True,
                text=True,
                timeout=num_games * 60  # 60 seconds per game max
            )
            
            # Parse output for win rate
            # Look for "Wins:   X (YY.Y%)" format
            for line in result.stdout.split('\n'):
                if 'Wins:' in line or 'wins:' in line:
                    # Extract percentage from parentheses
                    import re
                    match = re.search(r'\((\d+\.?\d*)%\)', line)
                    if match:
                        try:
                            return float(match.group(1)) / 100.0
                        except:
                            pass
                # Also support legacy "Win rate:" format
                elif 'Win rate:' in line or 'win rate:' in line:
                    # Extract percentage
                    parts = line.split(':')
                    if len(parts) >= 2:
                        pct_str = parts[1].strip().split()[0].replace('%', '')
                        try:
                            return float(pct_str) / 100.0
                        except:
                            pass
            
            # Fallback: try to find results file
            results_dir = Path('benchmark_results_live')
            if results_dir.exists():
                result_files = sorted(results_dir.glob('results_*.json'))
                if result_files:
                    with open(result_files[-1], 'r') as f:
                        data = json.load(f)
                        return data.get('win_rate', 0.0)
            
            print(f"  Warning: Could not parse win rate, assuming 0")
            return 0.0
            
        except subprocess.TimeoutExpired:
            print(f"  Warning: Benchmark timeout, assuming 0")
            return 0.0
        except Exception as e:
            print(f"  Error running benchmark: {e}")
            return 0.0
    
    def rebuild_snake(self):
        """Rebuild Go snake with new config"""
        print("  Rebuilding snake...")
        try:
            subprocess.run(['go', 'build', '-o', 'battlesnake', '.'], 
                         check=True, capture_output=True, timeout=120)
            return True
        except Exception as e:
            print(f"  Error building snake: {e}")
            return False
    
    def load_config(self):
        """Load current config.yaml"""
        with open(self.best_config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config):
        """Save config to config.yaml"""
        with open(self.best_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def git_commit_improvement(self, win_rate, improvement):
        """Commit improved weights to git repository"""
        try:
            # Configure git if not already configured
            subprocess.run(['git', 'config', '--get', 'user.name'], 
                         capture_output=True, check=False)
            result = subprocess.run(['git', 'config', '--get', 'user.name'], 
                                  capture_output=True, text=True)
            if not result.stdout.strip():
                subprocess.run(['git', 'config', 'user.name', 'Automated Training Bot'], check=False)
                subprocess.run(['git', 'config', 'user.email', 'training-bot@battlesnake.local'], check=False)
            
            # Stage config.yaml
            subprocess.run(['git', 'add', 'config.yaml'], check=True, capture_output=True)
            
            # Create commit message with improvement details
            commit_msg = (
                f"Automated training improvement: {win_rate:.2%} win rate (+{improvement:.2%})\n\n"
                f"Iteration: {self.state['iteration']}\n"
                f"Total games tested: {self.state['total_games']}\n"
                f"Total improvements: {self.state['improvements']}\n"
                f"Timestamp: {datetime.now().isoformat()}\n\n"
                f"[Automated commit by continuous training system]"
            )
            
            # Commit
            subprocess.run(['git', 'commit', '-m', commit_msg], 
                         check=True, capture_output=True)
            
            # Push to origin (if remote exists)
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, check=False)
            if result.returncode == 0:
                subprocess.run(['git', 'push', 'origin', 'HEAD'], 
                             check=True, capture_output=True, timeout=60)
                print(f"   ✓ Committed and pushed to git repository")
            else:
                print(f"   ✓ Committed to git repository (no remote to push)")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ⚠ Warning: Could not commit to git: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"   ⚠ Warning: Git push timeout (commit saved locally)")
            return False
        except Exception as e:
            print(f"   ⚠ Warning: Git commit failed: {e}")
            return False
    
    def random_perturbation(self, config, magnitude=0.1):
        """Apply random perturbation to weights"""
        import random
        
        new_config = config.copy()
        weights = new_config.get('weights', {})
        
        # List of weights to perturb
        weight_keys = [
            'space_weight', 'head_collision_weight', 'center_weight',
            'wall_penalty_weight', 'cutoff_weight', 'trap_moderate',
            'trap_severe', 'trap_critical', 'food_trap', 'pursuit_dist2',
            'pursuit_dist3', 'pursuit_dist4', 'pursuit_dist5'
        ]
        
        # Randomly select 3-5 weights to adjust
        num_to_adjust = random.randint(3, 5)
        keys_to_adjust = random.sample(weight_keys, num_to_adjust)
        
        for key in keys_to_adjust:
            if key in weights:
                current_val = weights[key]
                # Apply random change: -magnitude to +magnitude
                change = random.uniform(-magnitude, magnitude)
                new_val = current_val * (1 + change)
                # Clamp to reasonable range
                new_val = max(1, min(1000, new_val))
                weights[key] = int(new_val)
        
        new_config['weights'] = weights
        return new_config
    
    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        print("\n⚠ Shutdown signal received, saving checkpoint...")
        self.running = False
        self.save_checkpoint()
        print("✓ Checkpoint saved. Training can be resumed later.")
        sys.exit(0)
    
    def train(self):
        """Main training loop"""
        print("=" * 70)
        print("🚀 24/7 CONTINUOUS NEURAL NETWORK TRAINING")
        print("=" * 70)
        print(f"Config:")
        print(f"  - Games per iteration: {self.games_per_iteration}")
        print(f"  - Checkpoint interval: every {self.checkpoint_interval} iterations")
        print(f"  - Max iterations: {'unlimited' if self.max_iterations is None else self.max_iterations}")
        print(f"  - Min improvement: {self.min_improvement:.4f}")
        print(f"  - Results directory: {self.results_dir}")
        print("=" * 70)
        print()
        
        # Get baseline win rate if not set
        if self.state['best_win_rate'] == 0.0:
            print("📊 Establishing baseline win rate...")
            baseline_wr = self.run_benchmark(50)
            self.state['best_win_rate'] = baseline_wr
            self.state['best_config'] = self.load_config()
            print(f"✓ Baseline: {baseline_wr:.1%}\n")
            self.save_checkpoint()
        
        iteration = self.state['iteration']
        
        while self.running:
            if self.max_iterations and iteration >= self.max_iterations:
                print(f"\n✓ Reached max iterations ({self.max_iterations})")
                break
            
            iteration += 1
            self.state['iteration'] = iteration
            
            print(f"{'='*70}")
            print(f"📈 ITERATION {iteration}")
            print(f"{'='*70}")
            print(f"Best win rate so far: {self.state['best_win_rate']:.1%}")
            print(f"Total games tested: {self.state['total_games']}")
            print(f"Improvements found: {self.state['improvements']}")
            print(f"Last improvement: iteration {self.state['last_improvement_iteration']}")
            print()
            
            start_time = time.time()
            
            # Load current best config
            current_config = self.load_config()
            
            # Generate candidate config with random perturbation
            print("🔄 Generating candidate configuration...")
            candidate_config = self.random_perturbation(
                current_config, 
                magnitude=0.15  # 15% random adjustment
            )
            
            # Save candidate and rebuild
            self.save_config(candidate_config)
            if not self.rebuild_snake():
                print("❌ Build failed, skipping iteration\n")
                self.save_config(current_config)  # Restore previous config
                continue
            
            # Test candidate
            win_rate = self.run_benchmark(self.games_per_iteration)
            self.state['total_games'] += self.games_per_iteration
            
            duration = time.time() - start_time
            
            # Log iteration
            iteration_data = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'win_rate': win_rate,
                'best_win_rate': self.state['best_win_rate'],
                'improvement': win_rate - self.state['best_win_rate'],
                'games': self.games_per_iteration,
                'duration_seconds': int(duration),
                'config': candidate_config
            }
            self.log_iteration(iteration_data)
            
            # Check if improvement
            improvement = win_rate - self.state['best_win_rate']
            
            if improvement > self.min_improvement:
                print(f"\n✅ IMPROVEMENT FOUND! +{improvement:.1%}")
                print(f"   New best: {win_rate:.1%} (was {self.state['best_win_rate']:.1%})")
                self.state['best_win_rate'] = win_rate
                self.state['best_config'] = candidate_config
                self.state['improvements'] += 1
                self.state['last_improvement_iteration'] = iteration
                # Config already saved
                
                # Commit improvement to git
                print(f"   📝 Committing improvement to git...")
                self.git_commit_improvement(win_rate, improvement)
            else:
                print(f"\n⚪ No improvement: {win_rate:.1%} (best: {self.state['best_win_rate']:.1%})")
                # Restore best config
                self.save_config(current_config)
                self.rebuild_snake()
            
            print(f"⏱  Duration: {duration:.1f}s ({duration/self.games_per_iteration:.1f}s/game)")
            print()
            
            # Checkpoint periodically
            if iteration % self.checkpoint_interval == 0:
                print("💾 Saving checkpoint...")
                self.save_checkpoint()
                print("✓ Checkpoint saved\n")
            
            # Check for stagnation (no improvement in 100 iterations)
            if iteration - self.state['last_improvement_iteration'] >= 100:
                print("⚠ WARNING: No improvement in 100 iterations")
                print("  Consider adjusting perturbation magnitude or stopping training")
                print()
        
        # Final checkpoint
        print("\n" + "="*70)
        print("🏁 TRAINING COMPLETE")
        print("="*70)
        print(f"Total iterations: {iteration}")
        print(f"Total games: {self.state['total_games']}")
        print(f"Total improvements: {self.state['improvements']}")
        print(f"Best win rate: {self.state['best_win_rate']:.1%}")
        print(f"Results saved to: {self.results_dir}")
        print("="*70)
        
        self.save_checkpoint()


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='24/7 Continuous NN Training')
    parser.add_argument('--games', type=int, default=30,
                       help='Games per iteration (default: 30)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N iterations (default: 10)')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum iterations (default: unlimited)')
    parser.add_argument('--min-improvement', type=float, default=0.001,
                       help='Minimum improvement to keep config (default: 0.001 = 0.1%%)')
    
    args = parser.parse_args()
    
    trainer = ContinuousTrainer(
        games_per_iteration=args.games,
        checkpoint_interval=args.checkpoint_interval,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
