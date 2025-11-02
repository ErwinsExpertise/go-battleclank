#!/usr/bin/env python3
"""
24/7 Continuous Neural Network Training for Battlesnake Weight Optimization
Enhanced with LLM-based intelligent weight suggestion and full config optimization

NEW FEATURES:
- Neural Network and LLM Integration: NN learns winning patterns and informs LLM suggestions
- Change History Tracking: Prevents repeated failed attempts
- Multi-GPU Parallel Training: Maximize utilization on 8x A100 GPU servers
- Parallel Configuration Testing: Test multiple configs simultaneously for faster convergence

Runs indefinitely, automatically managing checkpoints and improvements
"""

import subprocess
import json
import yaml
import os
import sys
import time
import signal
import copy
import random
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# LLM and Neural Network imports (with fallback for missing dependencies)
try:
    import torch
    import torch.nn as nn
    import numpy as np
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural network optimization disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    transformers.logging.set_verbosity_error()  # Reduce noise
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: Transformers not available. LLM-guided optimization disabled.")


# Define classes only if dependencies are available
if PYTORCH_AVAILABLE:
    class ConfigPatternNetwork(nn.Module):
        """Neural network that learns patterns in successful configurations"""
        
        def __init__(self, input_size=36, hidden_size=128):
            super(ConfigPatternNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, input_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
            # Store learned patterns for analysis
            self.winning_patterns = []
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
        
        def extract_winning_patterns(self):
            """Extract and analyze winning configuration patterns learned by the network"""
            if not self.winning_patterns:
                return None
            
            # Analyze recent successful patterns
            recent_winners = self.winning_patterns[-10:]
            
            if not recent_winners:
                return None
            
            # Convert to numpy for analysis
            patterns_array = np.array(recent_winners)
            
            # Calculate statistics
            mean_values = np.mean(patterns_array, axis=0)
            std_values = np.std(patterns_array, axis=0)
            
            # Identify most consistent (low variance) winning parameters
            consistency_scores = 1 / (std_values + 0.01)  # Add small epsilon to avoid division by zero
            top_indices = np.argsort(consistency_scores)[-5:]  # Top 5 most consistent
            
            # Map indices to parameter names (simplified mapping)
            param_names = [
                'space', 'head_collision', 'center_control', 'wall_penalty', 'cutoff', 'food',
                'trap_moderate', 'trap_severe', 'trap_critical', 'food_trap', 'food_trap_threshold',
                'pursuit_2', 'pursuit_3', 'pursuit_4', 'pursuit_5',
                'trapping_weight', 'trapping_cutoff', 'trapped_ratio',
                'urgency_critical', 'urgency_low', 'urgency_normal',
                'late_game_caution', 'late_game_threshold',
                'hybrid_health', 'hybrid_enemies', 'hybrid_space', 'hybrid_depth', 'hybrid_mcts', 'hybrid_timeout',
                'search_nodes', 'search_depth',
                'opt_lr', 'opt_discount', 'opt_exploration', 'opt_batch', 'opt_episodes'
            ]
            
            # Build insight string
            insights = []
            for idx in top_indices:
                if idx < len(param_names):
                    param_name = param_names[idx]
                    mean_val = mean_values[idx]
                    insights.append(f"- {param_name}: consistently ~{mean_val:.2f} in wins")
            
            return "\n".join(insights) if insights else None
        
        def record_winning_config(self, config_vector):
            """Record a winning configuration for pattern analysis"""
            self.winning_patterns.append(config_vector.tolist() if hasattr(config_vector, 'tolist') else config_vector)
            # Keep only recent winners (last 50)
            if len(self.winning_patterns) > 50:
                self.winning_patterns = self.winning_patterns[-50:]
else:
    ConfigPatternNetwork = None


class LLMWeightAdvisor:
    """Lightweight LLM-based advisor for intelligent weight adjustments"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.change_history = []  # Track all attempted changes to avoid repetition
        if not PYTORCH_AVAILABLE:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try to load a lightweight model (TinyLlama is ~1GB, good for A100)
        if LLM_AVAILABLE:
            try:
                print("🤖 Loading lightweight LLM (TinyLlama-1.1B)...")
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✓ LLM loaded successfully on {self.device}")
            except Exception as e:
                print(f"⚠ Could not load LLM: {e}")
                print("  Falling back to random perturbation")
                self.model = None
    
    def suggest_adjustments(self, config, recent_history, current_win_rate, best_win_rate, nn_patterns=None):
        """Use LLM to suggest intelligent weight adjustments based on training history and NN patterns"""
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            # Build context for the LLM including NN patterns and change history
            prompt = self._build_prompt(config, recent_history, current_win_rate, best_win_rate, nn_patterns)
            
            # Generate suggestion
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the response to extract weight adjustments
            suggestions = self._parse_suggestions(response)
            
            # Record this suggestion in change history to avoid repetition
            self._record_change_attempt(suggestions)
            
            return suggestions
            
        except Exception as e:
            print(f"  ⚠ LLM suggestion failed: {e}")
            return None
    
    def _build_prompt(self, config, recent_history, current_win_rate, best_win_rate, nn_patterns=None):
        """Build a prompt for the LLM with training context, NN patterns, and change history"""
        # Summarize recent performance
        recent_improvements = [h for h in recent_history[-10:] if h.get('improvement', 0) > 0]
        recent_declines = [h for h in recent_history[-10:] if h.get('improvement', 0) <= 0]
        
        # Build change history summary
        change_summary = self._summarize_change_history()
        
        # Build NN pattern insights
        pattern_insights = ""
        if nn_patterns:
            pattern_insights = f"\nNeural Network Insights:\n{nn_patterns}\n"
        
        prompt = f"""<|system|>
You are an AI expert in Battlesnake game optimization. Analyze training results and suggest parameter adjustments.
</s>
<|user|>
Current battlesnake win rate: {current_win_rate:.1%}
Best win rate so far: {best_win_rate:.1%}

Recent performance:
- Last 10 iterations: {len(recent_improvements)} improvements, {len(recent_declines)} declines
- Stagnating: {'Yes' if len(recent_improvements) == 0 else 'No'}
{pattern_insights}
Previously tried adjustments (avoid repeating):
{change_summary}

Key parameters:
- space weight: {config.get('weights', {}).get('space', 5.0)}
- head_collision: {config.get('weights', {}).get('head_collision', 500.0)}
- food urgency critical: {config.get('food_urgency', {}).get('critical', 1.8)}
- trap critical: {config.get('traps', {}).get('critical', 600.0)}

Based on this data, which 3-5 parameters should be adjusted and by how much (+/-10-20%)? Consider:
1. If performance is declining, revert recent changes
2. If stagnating, try larger adjustments
3. Balance offensive (food) and defensive (traps) parameters
4. Avoid parameters that were recently tried without success
5. Follow neural network insights on winning patterns
6. Respond with parameter names and adjustment directions only.
</s>
<|assistant|>
Based on analysis, I suggest adjusting:"""
        
        return prompt
    
    def _parse_suggestions(self, response):
        """Extract parameter adjustment suggestions from LLM response"""
        suggestions = []
        
        # Simple parsing - look for parameter names and increase/decrease keywords
        keywords = {
            'increase': 0.15, 'raise': 0.15, 'boost': 0.15, 'higher': 0.15, 'up': 0.15,
            'decrease': -0.15, 'lower': -0.15, 'reduce': -0.15, 'down': -0.15
        }
        
        param_names = [
            'space', 'head_collision', 'center_control', 'wall_penalty', 'cutoff', 'food',
            'moderate', 'severe', 'critical', 'food_trap', 
            'distance_2', 'distance_3', 'distance_4', 'distance_5',
            'food_urgency', 'caution', 'threshold'
        ]
        
        response_lower = response.lower()
        
        for param in param_names:
            if param in response_lower:
                for keyword, adjustment in keywords.items():
                    if keyword in response_lower:
                        suggestions.append({'param': param, 'adjustment': adjustment})
                        break
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _record_change_attempt(self, suggestions):
        """Record attempted changes to avoid repetition"""
        if not suggestions:
            return
        
        for suggestion in suggestions:
            param = suggestion.get('param', '')
            adjustment = suggestion.get('adjustment', 0)
            self.change_history.append({
                'param': param,
                'adjustment': adjustment,
                'timestamp': datetime.now().isoformat()
            })
        
        # Keep only recent history (last 100 attempts)
        if len(self.change_history) > 100:
            self.change_history = self.change_history[-100:]
    
    def _summarize_change_history(self):
        """Summarize recent change attempts to avoid repetition"""
        if not self.change_history:
            return "None yet"
        
        # Get last 20 changes
        recent = self.change_history[-20:]
        
        # Count by parameter
        param_counts = {}
        for change in recent:
            param = change['param']
            param_counts[param] = param_counts.get(param, 0) + 1
        
        # Format summary
        if not param_counts:
            return "None yet"
        
        summary_parts = []
        for param, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            summary_parts.append(f"{param} ({count}x)")
        
        return "Recently adjusted: " + ", ".join(summary_parts)


class ContinuousTrainer:
    """Manages 24/7 continuous training with automatic checkpointing and recovery
    Enhanced with LLM-guided optimization and neural network pattern recognition"""
    
    def __init__(self, 
                 games_per_iteration=30,
                 checkpoint_interval=10,
                 max_iterations=None,
                 min_improvement=0.001,
                 use_llm=True,
                 use_neural_net=True,
                 parallel_configs=1):
        self.games_per_iteration = games_per_iteration
        self.checkpoint_interval = checkpoint_interval
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.use_llm = use_llm and LLM_AVAILABLE
        self.use_neural_net = use_neural_net and PYTORCH_AVAILABLE
        self.parallel_configs = parallel_configs  # Number of configs to test in parallel
        
        self.results_dir = Path("nn_training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.checkpoint_file = self.results_dir / "checkpoint.json"
        self.best_config_file = Path("config.yaml")
        self.global_log_file = self.results_dir / "training_log.jsonl"
        
        # Initialize LLM advisor
        self.llm_advisor = None
        if self.use_llm:
            try:
                self.llm_advisor = LLMWeightAdvisor()
                print("✓ LLM advisor initialized")
            except Exception as e:
                print(f"⚠ Could not initialize LLM: {e}")
                self.use_llm = False
        
        # Initialize neural network for pattern recognition
        self.pattern_network = None
        self.pattern_optimizer = None
        if self.use_neural_net:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pattern_network = ConfigPatternNetwork().to(device)
                self.pattern_optimizer = torch.optim.Adam(self.pattern_network.parameters(), lr=0.001)
                print(f"✓ Pattern recognition network initialized on {device}")
            except Exception as e:
                print(f"⚠ Could not initialize neural network: {e}")
                self.use_neural_net = False
        
        # Load or initialize state
        self.state = self.load_checkpoint()
        
        # Training history for LLM context
        self.recent_history = []
        
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
    
    def run_parallel_benchmarks(self, configs, games_per_config):
        """Run benchmarks for multiple configurations in parallel to maximize GPU usage
        
        Args:
            configs: List of configuration dictionaries to test
            games_per_config: Number of games to run for each configuration
            
        Returns:
            List of (config, win_rate) tuples
        """
        if not PYTORCH_AVAILABLE:
            # Fallback to sequential if PyTorch not available
            results = []
            for config in configs:
                self.save_config(config)
                self.rebuild_snake()
                win_rate = self.run_benchmark(games_per_config)
                results.append((config, win_rate))
            return results
        
        try:
            import concurrent.futures
            import tempfile
            import shutil
            
            print(f"  🚀 Running {len(configs)} configurations in parallel across GPUs...")
            
            def test_single_config(args):
                config, config_id, temp_dir = args
                try:
                    # Create temporary workspace for this config
                    workspace = Path(temp_dir) / f"workspace_{config_id}"
                    workspace.mkdir(exist_ok=True, parents=True)
                    
                    # Copy necessary files
                    config_file = workspace / "config.yaml"
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    # Build in workspace
                    build_result = subprocess.run(
                        ['go', 'build', '-o', str(workspace / 'battlesnake'), '.'],
                        cwd=Path.cwd(),
                        capture_output=True,
                        timeout=120
                    )
                    
                    if build_result.returncode != 0:
                        return (config, 0.0)
                    
                    # Set GPU for this worker (round-robin across available GPUs)
                    gpu_id = config_id % torch.cuda.device_count() if torch.cuda.is_available() else 0
                    env = os.environ.copy()
                    if torch.cuda.is_available():
                        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    
                    # Run benchmark
                    result = subprocess.run(
                        ['python3', 'tools/run_benchmark.py', str(games_per_config)],
                        capture_output=True,
                        text=True,
                        timeout=games_per_config * 60,
                        env=env
                    )
                    
                    # Parse win rate
                    win_rate = 0.0
                    for line in result.stdout.split('\n'):
                        if 'Wins:' in line or 'wins:' in line:
                            import re
                            match = re.search(r'\((\d+\.?\d*)%\)', line)
                            if match:
                                win_rate = float(match.group(1)) / 100.0
                                break
                    
                    return (config, win_rate)
                    
                except Exception as e:
                    print(f"  ⚠ Parallel benchmark error for config {config_id}: {e}")
                    return (config, 0.0)
                finally:
                    # Cleanup workspace
                    if workspace.exists():
                        shutil.rmtree(workspace, ignore_errors=True)
            
            # Create temp directory for parallel workspaces
            temp_base = Path(tempfile.mkdtemp(prefix="parallel_training_"))
            
            # Prepare arguments
            args_list = [(config, i, str(temp_base)) for i, config in enumerate(configs)]
            
            # Determine number of workers (one per GPU, or 4 for CPU)
            num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 4
            num_workers = min(num_workers, len(configs))
            
            print(f"  Using {num_workers} parallel workers")
            
            # Run in parallel
            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(test_single_config, args) for args in args_list]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        print(f"  ✓ Config completed: {result[1]:.1%} win rate")
                    except Exception as e:
                        print(f"  ⚠ Parallel execution error: {e}")
            
            # Cleanup
            shutil.rmtree(temp_base, ignore_errors=True)
            
            return results
            
        except Exception as e:
            print(f"  ⚠ Parallel benchmark setup failed: {e}")
            print(f"  Falling back to sequential execution")
            # Fallback to sequential
            results = []
            for config in configs:
                self.save_config(config)
                self.rebuild_snake()
                win_rate = self.run_benchmark(games_per_config)
                results.append((config, win_rate))
            return results
    
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
            
            # Check if there are actually changes to commit
            result = subprocess.run(['git', 'diff', '--cached', '--quiet'], 
                                  capture_output=True, check=False)
            if result.returncode == 0:
                # No changes staged
                print(f"   ⚠ Warning: No changes to commit (config.yaml unchanged)")
                return False
            
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
    
    def get_all_tunable_params(self, config):
        """Build comprehensive list of ALL tunable parameters in config"""
        tunable_params = []
        
        # Weights section
        weights = config.get('weights', {})
        for key in weights.keys():
            tunable_params.append(('weights', key, 0.1, 1000.0))  # (section, key, min, max)
        
        # Pursuit section
        pursuit = config.get('pursuit', {})
        for key in pursuit.keys():
            tunable_params.append(('pursuit', key, 1.0, 500.0))
        
        # Traps section
        traps = config.get('traps', {})
        for key, val in traps.items():
            if isinstance(val, (int, float)) and key != 'food_trap_threshold':
                tunable_params.append(('traps', key, 1.0, 2000.0))
            elif key == 'food_trap_threshold':
                tunable_params.append(('traps', key, 0.5, 0.95))
        
        # Food urgency section
        food_urgency = config.get('food_urgency', {})
        for key in food_urgency.keys():
            tunable_params.append(('food_urgency', key, 1.0, 3.0))
        
        # Trapping section
        trapping = config.get('trapping', {})
        for key, val in trapping.items():
            if isinstance(val, (int, float)):
                if 'threshold' in key or 'ratio' in key:
                    tunable_params.append(('trapping', key, 0.1, 0.9))
                else:
                    tunable_params.append(('trapping', key, 50.0, 1000.0))
        
        # Late game section
        late_game = config.get('late_game', {})
        for key, val in late_game.items():
            if isinstance(val, (int, float)):
                if 'multiplier' in key:
                    tunable_params.append(('late_game', key, 1.0, 2.0))
                elif 'threshold' in key:
                    tunable_params.append(('late_game', key, 50, 300))
        
        # Hybrid section (numeric only)
        hybrid = config.get('hybrid', {})
        for key, val in hybrid.items():
            if isinstance(val, (int, float)):
                if 'health' in key:
                    tunable_params.append(('hybrid', key, 10, 50))
                elif 'depth' in key:
                    tunable_params.append(('hybrid', key, 1, 5))
                elif 'iterations' in key:
                    tunable_params.append(('hybrid', key, 50, 200))
                elif 'timeout' in key:
                    tunable_params.append(('hybrid', key, 100, 500))
                elif 'ratio' in key:
                    tunable_params.append(('hybrid', key, 1.0, 5.0))
                elif 'enemies' in key:
                    tunable_params.append(('hybrid', key, 1, 4))
        
        # Search section (numeric only)
        search = config.get('search', {})
        for key, val in search.items():
            if isinstance(val, (int, float)):
                if 'nodes' in key:
                    tunable_params.append(('search', key, 100, 1000))
                elif 'depth' in key:
                    tunable_params.append(('search', key, 50, 200))
        
        # Optimization section (numeric only)
        optimization = config.get('optimization', {})
        for key, val in optimization.items():
            if isinstance(val, (int, float)):
                if 'rate' in key or 'factor' in key:
                    tunable_params.append(('optimization', key, 0.001, 0.1))
                elif 'batch' in key:
                    tunable_params.append(('optimization', key, 16, 128))
                elif 'episodes' in key:
                    tunable_params.append(('optimization', key, 500, 5000))
        
        return tunable_params
    
    def apply_adjustment(self, config, section, key, change_magnitude, min_val, max_val):
        """Apply adjustment to a specific parameter with bounds checking"""
        section_data = config.get(section, {})
        if key not in section_data:
            return
        
        current_val = section_data[key]
        if not isinstance(current_val, (int, float)):
            return
        
        # Apply change
        new_val = current_val * (1 + change_magnitude)
        
        # Clamp to range
        new_val = max(min_val, min(max_val, new_val))
        
        # Round appropriately
        if isinstance(current_val, int) and max_val < 10:
            new_val = int(round(new_val))
        elif isinstance(current_val, int):
            new_val = int(round(new_val))
        elif max_val < 5:
            new_val = round(new_val, 2)
        else:
            new_val = round(new_val, 1)
        
        section_data[key] = new_val
        config[section] = section_data
    
    def intelligent_perturbation(self, config, magnitude=0.15):
        """Apply intelligent perturbation using LLM guidance with NN pattern insights"""
        new_config = copy.deepcopy(config)
        
        # Extract winning patterns from neural network if available
        nn_patterns = None
        if self.use_neural_net and self.pattern_network is not None:
            try:
                nn_patterns = self.pattern_network.extract_winning_patterns()
                if nn_patterns:
                    print(f"  🧠 Neural network identified winning patterns")
            except Exception as e:
                print(f"  ⚠ NN pattern extraction warning: {e}")
        
        # Get LLM suggestions if available, passing NN patterns
        llm_suggestions = None
        if self.use_llm and self.llm_advisor is not None:
            try:
                llm_suggestions = self.llm_advisor.suggest_adjustments(
                    config, 
                    self.recent_history,
                    self.state.get('best_win_rate', 0.0),
                    self.state.get('best_win_rate', 0.0),
                    nn_patterns
                )
                if llm_suggestions:
                    print(f"  🤖 LLM suggested {len(llm_suggestions)} parameter adjustments")
            except Exception as e:
                print(f"  ⚠ LLM suggestion error: {e}")
        
        # Get all tunable parameters
        tunable_params = self.get_all_tunable_params(new_config)
        
        if not tunable_params:
            return new_config
        
        # Decide which parameters to adjust
        if llm_suggestions and len(llm_suggestions) > 0:
            # Use LLM-guided selection
            params_to_adjust = []
            for suggestion in llm_suggestions:
                param_name = suggestion['param']
                adjustment = suggestion['adjustment']
                
                # Find matching parameter
                for section, key, min_val, max_val in tunable_params:
                    if param_name in key.lower() or key.lower() in param_name:
                        params_to_adjust.append((section, key, adjustment, min_val, max_val))
                        break
            
            # Fill with random if needed
            while len(params_to_adjust) < 3:
                section, key, min_val, max_val = random.choice(tunable_params)
                adjustment = random.uniform(-magnitude, magnitude)
                params_to_adjust.append((section, key, adjustment, min_val, max_val))
        else:
            # Random selection (fallback)
            num_to_adjust = min(random.randint(3, 7), len(tunable_params))
            selected = random.sample(tunable_params, num_to_adjust)
            params_to_adjust = [
                (section, key, random.uniform(-magnitude, magnitude), min_val, max_val)
                for section, key, min_val, max_val in selected
            ]
        
        # Apply adjustments
        for section, key, change_magnitude, min_val, max_val in params_to_adjust:
            self.apply_adjustment(new_config, section, key, change_magnitude, min_val, max_val)
        
        return new_config
    
    def random_perturbation(self, config, magnitude=0.1):
        """Apply perturbation to weights - now uses intelligent method"""
        return self.intelligent_perturbation(config, magnitude)
    
    def _config_to_vector(self, config):
        """Convert configuration to a vector for neural network processing"""
        try:
            vector = []
            # Weights section (6 params)
            weights = config.get('weights', {})
            vector.extend([
                weights.get('space', 5.0),
                weights.get('head_collision', 500.0),
                weights.get('center_control', 2.0),
                weights.get('wall_penalty', 5.0),
                weights.get('cutoff', 100.0),
                weights.get('food', 3.0)
            ])
            # Traps section (5 params)
            traps = config.get('traps', {})
            vector.extend([
                traps.get('moderate', 200.0),
                traps.get('severe', 400.0),
                traps.get('critical', 600.0),
                traps.get('food_trap', 800.0),
                traps.get('food_trap_threshold', 0.8)
            ])
            # Pursuit section (4 params)
            pursuit = config.get('pursuit', {})
            vector.extend([
                pursuit.get('distance_2', 100.0),
                pursuit.get('distance_3', 50.0),
                pursuit.get('distance_4', 25.0),
                pursuit.get('distance_5', 10.0)
            ])
            # Trapping section (3 params)
            trapping = config.get('trapping', {})
            vector.extend([
                trapping.get('weight', 250.0),
                trapping.get('space_cutoff_threshold', 0.3),
                trapping.get('trapped_ratio', 0.5)
            ])
            # Food urgency section (3 params)
            food_urgency = config.get('food_urgency', {})
            vector.extend([
                food_urgency.get('critical', 1.8),
                food_urgency.get('low', 1.2),
                food_urgency.get('normal', 1.0)
            ])
            # Late game section (2 params)
            late_game = config.get('late_game', {})
            vector.extend([
                late_game.get('caution_multiplier', 1.5),
                late_game.get('turn_threshold', 150)
            ])
            # Hybrid section (6 params)
            hybrid = config.get('hybrid', {})
            vector.extend([
                hybrid.get('critical_health', 30),
                hybrid.get('critical_nearby_enemies', 2),
                hybrid.get('critical_space_ratio', 2.0),
                hybrid.get('lookahead_depth', 3),
                hybrid.get('mcts_iterations', 100),
                hybrid.get('mcts_timeout_ms', 300)
            ])
            # Search section (2 params)
            search = config.get('search', {})
            vector.extend([
                search.get('max_astar_nodes', 500),
                search.get('max_depth', 100)
            ])
            # Optimization section (5 params)
            optimization = config.get('optimization', {})
            vector.extend([
                optimization.get('learning_rate', 0.01),
                optimization.get('discount_factor', 0.95),
                optimization.get('exploration_rate', 0.1),
                optimization.get('batch_size', 32),
                optimization.get('episodes', 1000)
            ])
            
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            print(f"  ⚠ Config to vector conversion error: {e}")
            return None
    
    def _train_pattern_network_batch(self, successful_configs):
        """Train pattern network on a batch of successful configurations"""
        if not PYTORCH_AVAILABLE or self.pattern_network is None:
            return
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Convert configs to vectors
            vectors = []
            for config_data in successful_configs:
                config = config_data.get('config', {})
                vec = self._config_to_vector(config)
                if vec is not None:
                    vectors.append(vec)
            
            if len(vectors) < 2:
                return
            
            # Create tensors
            X = torch.tensor(np.array(vectors), dtype=torch.float32).to(device)
            
            # Train autoencoder to learn patterns
            self.pattern_network.train()
            self.pattern_optimizer.zero_grad()
            
            # Forward pass
            reconstructed = self.pattern_network(X)
            
            # Loss: reconstruction error
            loss = nn.MSELoss()(reconstructed, X)
            
            # Backward pass
            loss.backward()
            self.pattern_optimizer.step()
            
            self.pattern_network.eval()
            
        except Exception as e:
            # Don't interrupt training for NN errors
            pass
    
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
        print(f"  - Parallel configs: {self.parallel_configs}")
        print(f"  - Results directory: {self.results_dir}")
        
        # GPU information
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n🎮 GPU Information:")
            print(f"  - Available GPUs: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  - GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            
            if self.parallel_configs > 1:
                utilization = min(100, (self.parallel_configs / gpu_count) * 100)
                print(f"  - Expected GPU utilization: ~{utilization:.0f}%")
            else:
                print(f"  - GPU utilization: Limited (use --parallel-configs for more)")
        
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
            
            # Parallel mode: Generate and test multiple candidates simultaneously
            if self.parallel_configs > 1:
                print(f"🔄 Generating {self.parallel_configs} candidate configurations for parallel testing...")
                
                # Generate multiple candidate configs
                candidate_configs = []
                for i in range(self.parallel_configs):
                    candidate = self.random_perturbation(
                        current_config, 
                        magnitude=0.15  # 15% random adjustment
                    )
                    candidate_configs.append(candidate)
                
                # Test all candidates in parallel
                results = self.run_parallel_benchmarks(candidate_configs, self.games_per_iteration)
                self.state['total_games'] += self.games_per_iteration * len(results)
                
                # Find best result
                best_result = max(results, key=lambda x: x[1])
                candidate_config, win_rate = best_result
                
                print(f"  Best of {len(results)} parallel configs: {win_rate:.1%}")
                
            else:
                # Sequential mode: Generate and test single candidate
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
            
            # Update recent history for LLM context
            self.recent_history.append(iteration_data)
            if len(self.recent_history) > 50:  # Keep last 50 iterations
                self.recent_history = self.recent_history[-50:]
            
            # Train pattern recognition network on successful configs
            if self.use_neural_net and self.pattern_network is not None and len(self.recent_history) >= 10:
                try:
                    # Extract config vectors from history
                    successful_configs = [h for h in self.recent_history if h.get('improvement', 0) > 0]
                    if len(successful_configs) > 0:
                        # Record winning configurations for pattern analysis
                        latest_success = successful_configs[-1]
                        config_data = latest_success.get('config', {})
                        
                        # Convert config to vector for NN
                        config_vector = self._config_to_vector(config_data)
                        if config_vector is not None:
                            self.pattern_network.record_winning_config(config_vector)
                            print(f"  🧠 Recorded winning pattern in neural network")
                        
                        # Optional: Train network on successful patterns (batched)
                        if len(successful_configs) >= 5:
                            self._train_pattern_network_batch(successful_configs[-5:])
                except Exception as e:
                    # Silently continue if NN training fails - don't interrupt main loop
                    if hasattr(e, '__class__'):
                        print(f"  ⚠ Neural network training warning: {e.__class__.__name__}")
                    pass
            
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
    
    parser = argparse.ArgumentParser(
        description='24/7 Continuous NN Training with LLM-Guided Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - LLM-guided intelligent weight selection (TinyLlama-1.1B)
  - Neural network pattern recognition from successful configs
  - NN-LLM integration: NN patterns inform LLM suggestions
  - Change history tracking to avoid repeated failed attempts
  - Support for ALL config.yaml parameters (30+ tunable parameters)
  - Multi-GPU parallel training (maximize 8x A100 GPU utilization)
  - Parallel configuration testing for faster convergence

Examples:
  # Basic usage (sequential)
  python3 continuous_training.py
  
  # Maximize A100 GPU usage with parallel training
  python3 continuous_training.py --parallel-configs 8
  
  # Disable LLM for faster iterations
  python3 continuous_training.py --no-llm
  
  # Full power: LLM + NN + 8 parallel configs on A100
  python3 continuous_training.py --use-llm --use-neural-net --parallel-configs 8
        """
    )
    parser.add_argument('--games', type=int, default=30,
                       help='Games per iteration (default: 30)')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save checkpoint every N iterations (default: 10)')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum iterations (default: unlimited)')
    parser.add_argument('--min-improvement', type=float, default=0.001,
                       help='Minimum improvement to keep config (default: 0.001 = 0.1%%)')
    parser.add_argument('--use-llm', action='store_true', default=True,
                       help='Use LLM for intelligent weight suggestions (default: True)')
    parser.add_argument('--no-llm', action='store_false', dest='use_llm',
                       help='Disable LLM guidance')
    parser.add_argument('--use-neural-net', action='store_true', default=True,
                       help='Use neural network for pattern recognition (default: True)')
    parser.add_argument('--no-neural-net', action='store_false', dest='use_neural_net',
                       help='Disable neural network pattern recognition')
    parser.add_argument('--parallel-configs', type=int, default=1,
                       help='Number of configurations to test in parallel (default: 1). Use 4-8 for A100 servers.')
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  LLM-Enhanced Continuous Training for Battlesnake             ║
    ║  Intelligent Weight Optimization with Neural Networks         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    if args.use_llm and not LLM_AVAILABLE:
        print("⚠ Warning: LLM requested but transformers not installed")
        print("  Install with: pip install transformers accelerate")
        print("  Falling back to random perturbation\n")
    
    if args.use_neural_net and not PYTORCH_AVAILABLE:
        print("⚠ Warning: Neural network requested but PyTorch not installed")
        print("  Install with: pip install torch")
        print("  Falling back to basic optimization\n")
    
    trainer = ContinuousTrainer(
        games_per_iteration=args.games,
        checkpoint_interval=args.checkpoint_interval,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement,
        use_llm=args.use_llm,
        use_neural_net=args.use_neural_net,
        parallel_configs=args.parallel_configs
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
