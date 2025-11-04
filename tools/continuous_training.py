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
import re
import concurrent.futures
import tempfile
import shutil
import fcntl
import requests
from datetime import datetime
from pathlib import Path

# Constants for pattern recognition and history tracking
EPSILON = 0.01  # Small value to avoid division by zero in pattern analysis
TOP_CONSISTENT_PARAMS = 5  # Number of most consistent parameters to identify
MAX_WINNING_PATTERNS = 50  # Maximum number of winning configs to keep in memory
MAX_CHANGE_HISTORY = 100  # Maximum number of change attempts to track

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
    # Configuration vector size: calculated from config structure
    # This must match the number of parameters extracted in ContinuousTrainer._config_to_vector()
    # weights(9) + traps(12) + pursuit(4) + trapping(3) + food_urgency(3) + food_weights(13) + late_game(2) + hybrid(6) + search(2) + optimization(5) + tactics(7) + emergency_wall_escape(7) = 73
    CONFIG_VECTOR_SIZE = 73
    
    class ConfigPatternNetwork(nn.Module):
        """Neural network that learns patterns in successful configurations"""
        
        def __init__(self, input_size=CONFIG_VECTOR_SIZE, hidden_size=128):
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
            consistency_scores = 1 / (std_values + EPSILON)  # Add small epsilon to avoid division by zero
            top_indices = np.argsort(consistency_scores)[-TOP_CONSISTENT_PARAMS:]  # Top most consistent
            
            # Parameter names in order matching _config_to_vector()
            # This must stay in sync with _config_to_vector() method
            param_names = self._get_param_names()
            
            if len(param_names) != CONFIG_VECTOR_SIZE:
                # Safety check: if sizes don't match, return generic insights
                return f"Pattern analysis: {len(recent_winners)} winning configs analyzed"
            
            # Build insight string
            insights = []
            for idx in top_indices:
                if idx < len(param_names):
                    param_name = param_names[idx]
                    mean_val = mean_values[idx]
                    insights.append(f"- {param_name}: consistently ~{mean_val:.2f} in wins")
            
            return "\n".join(insights) if insights else None
        
        @staticmethod
        def _get_param_names():
            """Get parameter names in order matching _config_to_vector()
            
            This centralizes the parameter name list to ensure consistency.
            Must be kept in sync with ContinuousTrainer._config_to_vector()
            """
            return [
                # Weights section (6)
                'space', 'head_collision', 'center_control', 'wall_penalty', 'cutoff', 'food',
                # Traps section (5)
                'trap_moderate', 'trap_severe', 'trap_critical', 'food_trap', 'food_trap_threshold',
                # Pursuit section (4)
                'pursuit_2', 'pursuit_3', 'pursuit_4', 'pursuit_5',
                # Trapping section (3)
                'trapping_weight', 'trapping_cutoff', 'trapped_ratio',
                # Food urgency section (3)
                'urgency_critical', 'urgency_low', 'urgency_normal',
                # Late game section (2)
                'late_game_caution', 'late_game_threshold',
                # Hybrid section (6)
                'hybrid_health', 'hybrid_enemies', 'hybrid_space', 'hybrid_depth', 'hybrid_mcts', 'hybrid_timeout',
                # Search section (2)
                'search_nodes', 'search_depth',
                # Optimization section (5)
                'opt_lr', 'opt_discount', 'opt_exploration', 'opt_batch', 'opt_episodes',
                # Tactics section (7)
                'inward_trap_weight', 'inward_trap_min_length', 'aggressive_space_weight',
                'aggressive_space_threshold', 'predictive_avoidance', 'energy_conservation', 'wall_hugging',
                # Emergency wall escape section (7)
                'emergency_min_distance', 'emergency_max_distance', 'emergency_turn_bonus',
                'emergency_close_bonus', 'emergency_away_penalty', 'emergency_close_threshold', 'emergency_coord_tolerance'
            ]
        
        def record_winning_config(self, config_vector):
            """Record a winning configuration for pattern analysis"""
            self.winning_patterns.append(config_vector.tolist() if hasattr(config_vector, 'tolist') else config_vector)
            # Keep only recent winners
            if len(self.winning_patterns) > MAX_WINNING_PATTERNS:
                self.winning_patterns = self.winning_patterns[-MAX_WINNING_PATTERNS:]
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
        
        # Load a more powerful model optimized for 8x A100 GPU setup
        # Mistral-7B is highly capable and efficient on A100s
        if LLM_AVAILABLE:
            try:
                print("🤖 Loading Mistral-7B-Instruct (optimized for A100 GPUs)...")
                model_name = "mistralai/Mistral-7B-Instruct-v0.2"
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
                print(f"✓ Mistral-7B loaded successfully on {self.device}")
                print(f"  Model has superior reasoning and context understanding")
            except Exception as e:
                print(f"⚠ Could not load Mistral-7B: {e}")
                print("  Trying fallback to TinyLlama...")
                try:
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
                    print(f"✓ TinyLlama loaded as fallback on {self.device}")
                except Exception as e2:
                    print(f"⚠ Could not load any LLM: {e2}")
                    print("  Falling back to random perturbation")
                    self.model = None
    
    def suggest_adjustments(self, config, recent_history, current_win_rate, best_win_rate, nn_patterns=None, death_summary=""):
        """Use LLM to suggest intelligent weight adjustments based on training history and NN patterns"""
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            # Build context for the LLM including NN patterns and change history
            prompt = self._build_prompt(config, recent_history, current_win_rate, best_win_rate, nn_patterns, death_summary)
            
            # Generate suggestion
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased from 200 to 512 for more detailed analysis
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
    
    def _build_prompt(self, config, recent_history, current_win_rate, best_win_rate, nn_patterns=None, death_summary=""):
        """Build a prompt for the LLM with training context, NN patterns, death reasons, and change history"""
        # Summarize recent performance - increased from 10 to 30 for more context
        recent_improvements = [h for h in recent_history[-30:] if h.get('improvement', 0) > 0]
        recent_declines = [h for h in recent_history[-30:] if h.get('improvement', 0) <= 0]
        
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

Recent performance (last 30 iterations):
- Improvements: {len(recent_improvements)} ({len(recent_improvements)/30*100:.1f}%)
- Declines: {len(recent_declines)} ({len(recent_declines)/30*100:.1f}%)
- Stagnating: {'Yes' if len(recent_improvements) == 0 else 'No'}
{pattern_insights}
{death_summary}
Previously tried adjustments (avoid repeating):
{change_summary}

Key parameters:
- space weight: {config.get('weights', {}).get('space', 5.0)}
- head_collision: {config.get('weights', {}).get('head_collision', 500.0)}
- food urgency critical: {config.get('food_urgency', {}).get('critical', 1.8)}
- trap critical: {config.get('traps', {}).get('critical', 600.0)}
- food weight critical health: {config.get('food_weights', {}).get('critical_health', 500.0)}
- food weight healthy: {config.get('food_weights', {}).get('healthy_base', 80.0)}

Based on this data, which 3-5 parameters should be adjusted and by how much (+/-10-20%)? Consider:
1. If performance is declining, revert recent changes
2. If stagnating, try larger adjustments
3. Balance offensive (food) and defensive (traps) parameters
4. Avoid parameters that were recently tried without success
5. Follow neural network insights on winning patterns
6. Address common death reasons with targeted parameter changes
7. Respond with parameter names and adjustment directions only.
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
            'food_urgency', 'caution', 'threshold',
            'critical_health', 'low_health', 'medium_health', 'healthy_base', 'healthy_early_game'
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
        
        # Keep only recent history
        if len(self.change_history) > MAX_CHANGE_HISTORY:
            self.change_history = self.change_history[-MAX_CHANGE_HISTORY:]
    
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


def _test_single_config_worker(args):
    """Worker function for parallel config testing
    
    This function is at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple of (config, config_id, base_dir, games_per_config)
    
    Returns:
        Tuple of (config, win_rate)
    """
    config, config_id, base_dir, games_per_config = args
    workspace = None
    
    try:
        # Create temporary workspace for this config
        workspace = Path(base_dir) / f"workspace_{config_id}"
        workspace.mkdir(exist_ok=True, parents=True)
        
        # Save config to workspace
        config_file = workspace / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Build binary (no need to copy config, we'll use CLI flag)
        build_result = subprocess.run(
            ['go', 'build', '-o', str(workspace / 'battlesnake')],
            cwd=Path.cwd(),
            capture_output=True,
            timeout=120
        )
        
        if build_result.returncode != 0:
            return (config, 0.0)
        
        try:
            
            # Set GPU for this worker (round-robin across available GPUs)
            env = os.environ.copy()
            if PYTORCH_AVAILABLE:
                try:
                    import torch
                    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                        gpu_id = config_id % torch.cuda.device_count()
                        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                except ImportError:
                    pass  # torch not available, skip GPU assignment
            
            # Allocate unique ports for this worker to avoid conflicts
            # Base ports: 8000 (go), 8080 (rust)
            # Each worker gets: 8000 + (config_id * 100), 8080 + (config_id * 100)
            go_port = 8000 + (config_id * 100)
            rust_port = 8080 + (config_id * 100)
            
            # Check if servers are already running (started by start_training.sh)
            # If not, we need to start them ourselves
            servers_already_running = False
            try:
                # Quick check: try to connect to the Go server
                test_result = subprocess.run(
                    ['curl', '-s', '-f', f'http://localhost:{go_port}/', '-o', '/dev/null'],
                    capture_output=True,
                    timeout=2
                )
                servers_already_running = (test_result.returncode == 0)
            except:
                servers_already_running = False
            
            if not servers_already_running:
                # Start servers for this worker with custom config
                # Set the config path via environment variable for the server
                server_env = env.copy()
                server_env['BATTLESNAKE_CONFIG'] = str(config_file)
                
                # Start Go snake with config flag
                go_snake_process = subprocess.Popen(
                    [str(workspace / 'battlesnake'), '-config', str(config_file)],
                    env=server_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Start Rust baseline
                rust_snake_process = subprocess.Popen(
                    ['./baseline/target/release/baseline-snake'],
                    cwd=Path.cwd(),
                    env={'BIND_PORT': str(rust_port)},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Save PIDs for cleanup
                go_pid_file = Path(f'/tmp/battlesnake_go_{go_port}.pid')
                rust_pid_file = Path(f'/tmp/battlesnake_rust_{rust_port}.pid')
                go_pid_file.write_text(str(go_snake_process.pid))
                rust_pid_file.write_text(str(rust_snake_process.pid))
                
                # Wait a bit for servers to start
                time.sleep(3)
            
            try:
                # Run benchmark - use 'no-manage' to avoid starting/stopping servers
                result = subprocess.run(
                    ['python3', str(Path.cwd() / 'tools' / 'run_benchmark.py'), 
                     str(games_per_config), str(go_port), str(rust_port), 'no-manage'],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=games_per_config * 60,
                    env=env
                )
                
                # Parse win rate
                win_rate = 0.0
                for line in result.stdout.split('\n'):
                    if 'Wins:' in line or 'wins:' in line:
                        match = re.search(r'\((\d+\.?\d*)%\)', line)
                        if match:
                            win_rate = float(match.group(1)) / 100.0
                            break
                
                return (config, win_rate)
            
            finally:
                # Only stop servers if we started them ourselves
                if not servers_already_running:
                    stop_servers_script = Path.cwd() / 'tools' / 'stop_servers.sh'
                    subprocess.run(
                        [str(stop_servers_script), str(go_port), str(rust_port)],
                        capture_output=True,
                        timeout=30
                    )
        
        except Exception as inner_e:
            print(f"  ⚠ Error running benchmark for config {config_id}: {inner_e}")
            return (config, 0.0)
    
    except Exception as e:
        print(f"  ⚠ Parallel benchmark error for config {config_id}: {e}")
        return (config, 0.0)
    
    finally:
        # Cleanup workspace
        if workspace and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


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
                 parallel_configs=1,
                 server_url="http://localhost:8000",
                 benchmark_rounds=3,
                 test_algorithms=None):
        self.games_per_iteration = games_per_iteration
        self.checkpoint_interval = checkpoint_interval
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.use_llm = use_llm and LLM_AVAILABLE
        self.use_neural_net = use_neural_net and PYTORCH_AVAILABLE
        self.parallel_configs = parallel_configs  # Number of configs to test in parallel
        self.server_url = server_url  # Server URL for config reload endpoint
        self.benchmark_rounds = max(1, benchmark_rounds)  # Number of rounds to average for validation
        self.test_algorithms = test_algorithms or ['hybrid']  # Algorithms to test: hybrid, greedy, lookahead, mcts
        
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
    
    def run_benchmark(self, num_games, algorithm=None):
        """Run benchmark and return win rate with death analysis
        
        Args:
            num_games: Number of games to run
            algorithm: Algorithm to use (overrides config if specified)
        
        Returns:
            tuple: (win_rate, death_analysis_dict) where death_analysis contains reason/phase stats
        """
        algo_str = f" with {algorithm}" if algorithm else ""
        print(f"  Running {num_games}-game benchmark{algo_str}...")
        
        # Temporarily modify config if algorithm specified
        config_modified = False
        original_config = None
        if algorithm:
            try:
                config = self.load_config()
                original_config = copy.deepcopy(config)
                if 'search' not in config:
                    config['search'] = {}
                config['search']['algorithm'] = algorithm
                self.save_config(config)
                self.reload_config_via_endpoint() or self.rebuild_snake()
                config_modified = True
            except Exception as e:
                print(f"  ⚠ Could not set algorithm: {e}")
        
        try:
            result = subprocess.run(
                ['python3', 'tools/run_benchmark.py', str(num_games)],
                capture_output=True,
                text=True,
                timeout=num_games * 60  # 60 seconds per game max
            )
            
            # Initialize return values (will be populated from stdout or file)
            win_rate = 0.0  # Ratio format (0.0-1.0)
            death_analysis = {}
            
            # Parse output for win rate
            # Look for "Wins:   X (YY.Y%)" format
            for line in result.stdout.split('\n'):
                if 'Wins:' in line or 'wins:' in line:
                    # Extract percentage from parentheses
                    match = re.search(r'\((\d+\.?\d*)%\)', line)
                    if match:
                        try:
                            win_rate = float(match.group(1)) / 100.0
                        except:
                            pass
                # Also support legacy "Win rate:" format
                elif 'Win rate:' in line or 'win rate:' in line:
                    # Extract percentage
                    parts = line.split(':')
                    if len(parts) >= 2:
                        pct_str = parts[1].strip().split()[0].replace('%', '')
                        try:
                            win_rate = float(pct_str) / 100.0
                        except:
                            pass
            
            # Fallback: try to find results file for both win rate and death analysis
            results_dir = Path('benchmark_results_live')
            if results_dir.exists():
                result_files = sorted(results_dir.glob('results_*.json'))
                if result_files:
                    with open(result_files[-1], 'r') as f:
                        data = json.load(f)
                        if win_rate == 0.0:  # Only use file data if we didn't parse from stdout
                            # File stores win_rate as percentage (0-100), convert to ratio (0-1)
                            win_rate = data.get('results', {}).get('win_rate', 0.0) / 100.0
                        death_analysis = data.get('death_analysis', {})
            
            if win_rate == 0.0:
                print(f"  Warning: Could not parse win rate, assuming 0")
            
            return win_rate, death_analysis
            
        except subprocess.TimeoutExpired:
            print(f"  Warning: Benchmark timeout, assuming 0")
            return 0.0, {}
        except Exception as e:
            print(f"  Error running benchmark: {e}")
            return 0.0, {}
        finally:
            # Restore original config if modified
            if config_modified and original_config:
                try:
                    self.save_config(original_config)
                    self.reload_config_via_endpoint() or self.rebuild_snake()
                except:
                    pass
    
    def run_benchmark_rounds(self, num_games, rounds=1, algorithm=None):
        """Run multiple benchmark rounds and return statistics with aggregated death analysis
        
        Args:
            num_games: Number of games per round
            rounds: Number of rounds to run
            algorithm: Algorithm to use (overrides config if specified)
        
        Returns:
            dict: Statistics including mean, std, min, max, all round results, and aggregated death analysis
        """
        if rounds == 1:
            win_rate, death_analysis = self.run_benchmark(num_games, algorithm)
            return {
                'mean': win_rate,
                'std': 0.0,
                'min': win_rate,
                'max': win_rate,
                'rounds': [win_rate],
                'count': 1,
                'death_analysis': death_analysis
            }
        
        print(f"  📊 Running {rounds} rounds for validation (averaging)...")
        round_results = []
        all_death_analyses = []
        
        for round_num in range(1, rounds + 1):
            print(f"    Round {round_num}/{rounds}...", end=" ", flush=True)
            win_rate, death_analysis = self.run_benchmark(num_games, algorithm)
            round_results.append(win_rate)
            all_death_analyses.append(death_analysis)
            print(f"{win_rate:.1%}")
        
        if PYTORCH_AVAILABLE:
            mean_wr = np.mean(round_results)
            std_wr = np.std(round_results)
            min_wr = np.min(round_results)
            max_wr = np.max(round_results)
        else:
            mean_wr = sum(round_results) / len(round_results)
            std_wr = (sum((x - mean_wr) ** 2 for x in round_results) / len(round_results)) ** 0.5
            min_wr = min(round_results)
            max_wr = max(round_results)
        
        # Aggregate death analysis across all rounds
        aggregated_death_analysis = self._aggregate_death_analyses(all_death_analyses)
        
        print(f"  📊 Average: {mean_wr:.1%} (±{std_wr:.1%}), Range: [{min_wr:.1%}, {max_wr:.1%}]")
        
        return {
            'mean': mean_wr,
            'std': std_wr,
            'min': min_wr,
            'max': max_wr,
            'rounds': round_results,
            'count': rounds,
            'death_analysis': aggregated_death_analysis
        }
    
    def _aggregate_death_analyses(self, analyses):
        """Aggregate death analyses from multiple rounds
        
        Args:
            analyses: List of death_analysis dictionaries
            
        Returns:
            dict: Aggregated death analysis with combined statistics
        """
        aggregated = {
            'by_reason': {},
            'by_phase': {'early': 0, 'mid': 0, 'late': 0}
        }
        
        for analysis in analyses:
            if not analysis:
                continue
            
            # Aggregate by reason
            by_reason = analysis.get('by_reason', {})
            for reason, count in by_reason.items():
                aggregated['by_reason'][reason] = aggregated['by_reason'].get(reason, 0) + count
            
            # Aggregate by phase
            by_phase = analysis.get('by_phase', {})
            for phase, count in by_phase.items():
                if phase in aggregated['by_phase']:
                    aggregated['by_phase'][phase] += count
        
        return aggregated
    
    def _summarize_death_reasons(self, history_slice):
        """Summarize death reasons from recent iterations for LLM context
        
        Args:
            history_slice: List of recent iteration data dictionaries
            
        Returns:
            str: Formatted death reason summary
        """
        if not history_slice:
            return ""
        
        # Aggregate death reasons across iterations
        total_by_reason = {}
        total_by_phase = {'early': 0, 'mid': 0, 'late': 0}
        total_iterations = 0
        
        for iteration in history_slice:
            death_analysis = iteration.get('death_analysis', {})
            if not death_analysis:
                continue
            
            total_iterations += 1
            
            # Aggregate by reason
            by_reason = death_analysis.get('by_reason', {})
            for reason, count in by_reason.items():
                total_by_reason[reason] = total_by_reason.get(reason, 0) + count
            
            # Aggregate by phase
            by_phase = death_analysis.get('by_phase', {})
            for phase, count in by_phase.items():
                if phase in total_by_phase:
                    total_by_phase[phase] += count
        
        if not total_by_reason and sum(total_by_phase.values()) == 0:
            return ""
        
        # Build summary string
        summary_parts = ["\nDeath Reason Analysis (recent losses):"]
        
        # Top death reasons
        if total_by_reason:
            sorted_reasons = sorted(total_by_reason.items(), key=lambda x: x[1], reverse=True)[:3]
            total_deaths = sum(total_by_reason.values())
            
            for reason, count in sorted_reasons:
                pct = (count / total_deaths * 100) if total_deaths > 0 else 0
                summary_parts.append(f"- {reason}: {count} ({pct:.1f}%)")
        
        # Death by game phase
        total_phase_deaths = sum(total_by_phase.values())
        if total_phase_deaths > 0:
            summary_parts.append("\nBy game phase:")
            for phase in ['early', 'mid', 'late']:
                count = total_by_phase[phase]
                if count > 0:
                    pct = (count / total_phase_deaths * 100)
                    summary_parts.append(f"- {phase}: {count} ({pct:.1f}%)")
        
        return "\n".join(summary_parts)
    
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
                win_rate, _ = self.run_benchmark(games_per_config)  # Ignore death analysis in parallel mode
                results.append((config, win_rate))
            return results
        
        try:
            print(f"  🚀 Running {len(configs)} configurations in parallel across GPUs...")
            
            # Create temp directory for parallel workspaces
            temp_base = Path(tempfile.mkdtemp(prefix="parallel_training_"))
            
            # Prepare arguments for each worker
            args_list = [(config, i, str(temp_base), games_per_config) for i, config in enumerate(configs)]
            
            # Determine number of workers (one per GPU, or 4 for CPU)
            import torch
            num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 4
            num_workers = min(num_workers, len(configs))
            
            print(f"  Using {num_workers} parallel workers")
            
            # Run in parallel using module-level worker function
            results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_test_single_config_worker, args) for args in args_list]
                
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
                win_rate, _ = self.run_benchmark(games_per_config)  # Ignore death analysis in parallel mode
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
    
    def reload_config_via_endpoint(self):
        """Reload configuration via server's reload-config endpoint instead of rebuilding
        
        This is much faster than rebuilding and uses the server's hot-reload capability.
        """
        print("  Reloading config via server endpoint...")
        try:
            response = requests.post(f"{self.server_url}/reload-config", timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'ok':
                    print(f"  ✓ Config reloaded successfully via endpoint")
                    return True
                else:
                    print(f"  ⚠ Server returned non-ok status: {result}")
                    return False
            else:
                print(f"  ⚠ Server returned status code {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"  ⚠ Could not connect to server at {self.server_url}")
            print(f"  ℹ Falling back to rebuild")
            return False
        except Exception as e:
            print(f"  ⚠ Error reloading config via endpoint: {e}")
            print(f"  ℹ Falling back to rebuild")
            return False
    
    def load_config(self):
        """Load current config.yaml"""
        with open(self.best_config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config):
        """Save config to config.yaml"""
        with open(self.best_config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def test_algorithm_diversity(self, config, games_per_test=20):
        """Test configuration with different algorithm combinations
        
        Args:
            config: Configuration to test
            games_per_test: Number of games per algorithm
        
        Returns:
            dict: Results for each algorithm with win rates
        """
        print(f"  🔬 Testing algorithm diversity...")
        results = {}
        
        # Save current config temporarily
        self.save_config(config)
        self.reload_config_via_endpoint() or self.rebuild_snake()
        
        for algo in self.test_algorithms:
            print(f"    Testing {algo}...", end=" ", flush=True)
            win_rate, _ = self.run_benchmark(games_per_test, algorithm=algo)  # Ignore death analysis for algo testing
            results[algo] = win_rate
            print(f"{win_rate:.1%}")
        
        # Find best algorithm
        best_algo = max(results, key=results.get)
        best_wr = results[best_algo]
        
        print(f"  🏆 Best algorithm: {best_algo} at {best_wr:.1%}")
        
        return {
            'results': results,
            'best_algorithm': best_algo,
            'best_win_rate': best_wr
        }
    
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
            
            # Always push to remote
            subprocess.run(['git', 'push', 'origin', 'HEAD'], 
                         check=True, capture_output=True, timeout=60)
            print(f"   ✓ Committed and pushed to git repository")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ⚠ Warning: Could not commit/push to git: {e}")
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
        
        # Food weights section (new configurable food weights)
        food_weights = config.get('food_weights', {})
        for key, val in food_weights.items():
            if isinstance(val, (int, float)):
                if 'multiplier' in key or 'mult' in key or 'outmatched' in key:
                    # Multipliers should be in range 0.1 to 1.0
                    tunable_params.append(('food_weights', key, 0.1, 1.0))
                else:
                    # Absolute weight values
                    tunable_params.append(('food_weights', key, 50.0, 800.0))
        
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
        
        # Tactics section (numeric only)
        tactics = config.get('tactics', {})
        for key, val in tactics.items():
            if isinstance(val, (int, float)):
                if 'weight' in key:
                    tunable_params.append(('tactics', key, 0.0, 200.0))
                elif 'threshold' in key:
                    tunable_params.append(('tactics', key, 20, 100))
                elif 'min_enemy_length' in key:
                    tunable_params.append(('tactics', key, 3, 10))
        
        # Emergency wall escape section
        emergency = config.get('emergency_wall_escape', {})
        for key, val in emergency.items():
            if isinstance(val, (int, float)):
                if 'distance' in key and 'min' in key:
                    tunable_params.append(('emergency_wall_escape', key, 1, 3))
                elif 'distance' in key and 'max' in key:
                    tunable_params.append(('emergency_wall_escape', key, 3, 6))
                elif 'bonus' in key or 'penalty' in key:
                    tunable_params.append(('emergency_wall_escape', key, 50.0, 300.0))
                elif 'threshold' in key or 'tolerance' in key:
                    tunable_params.append(('emergency_wall_escape', key, 1, 5))
        
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
        
        # Get LLM suggestions if available, passing NN patterns and death summary
        llm_suggestions = None
        if self.use_llm and self.llm_advisor is not None:
            try:
                # Generate death summary for LLM context
                death_summary = self._summarize_death_reasons(self.recent_history[-30:])
                
                llm_suggestions = self.llm_advisor.suggest_adjustments(
                    config, 
                    self.recent_history,
                    self.state.get('best_win_rate', 0.0),
                    self.state.get('best_win_rate', 0.0),
                    nn_patterns,
                    death_summary
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
            # Weights section (9 params)
            weights = config.get('weights', {})
            vector.extend([
                weights.get('space', 5.0),
                weights.get('head_collision', 500.0),
                weights.get('center_control', 2.0),
                weights.get('wall_penalty', 5.0),
                weights.get('cutoff', 100.0),
                weights.get('food', 3.0),
                weights.get('space_base_multiplier', 1.5),
                weights.get('space_enemy_multiplier', 2.5),
                weights.get('space_healthy_multiplier', 1.2)
            ])
            # Traps section (12 params)
            traps = config.get('traps', {})
            vector.extend([
                traps.get('moderate', 200.0),
                traps.get('severe', 400.0),
                traps.get('critical', 600.0),
                traps.get('food_trap', 1200.0),
                traps.get('food_trap_threshold', 0.8),
                traps.get('food_trap_critical', 500.0),
                traps.get('food_trap_low', 800.0),
                traps.get('space_reduction_60', 1500.0),
                traps.get('space_reduction_50', 800.0),
                traps.get('space_reduction_ratio_60', 0.4),
                traps.get('space_reduction_ratio_50', 0.5),
                traps.get('space_reduction_min_base', 0.2)
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
            # Food weights section (13 params)
            food_weights = config.get('food_weights', {})
            vector.extend([
                food_weights.get('critical_health', 500.0),
                food_weights.get('critical_health_outmatched', 400.0),
                food_weights.get('low_health', 220.0),
                food_weights.get('low_health_outmatched', 180.0),
                food_weights.get('medium_health', 120.0),
                food_weights.get('medium_health_outmatched', 0.6),
                food_weights.get('healthy_base', 80.0),
                food_weights.get('healthy_early_game', 100.0),
                food_weights.get('healthy_outmatched', 0.5),
                food_weights.get('healthy_ceiling', 80),
                food_weights.get('healthy_ceiling_weight', 10.0),
                food_weights.get('healthy_multiplier', 0.5),
                food_weights.get('healthy_early_multiplier', 0.6)
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
            # Tactics section (7 params)
            tactics = config.get('tactics', {})
            vector.extend([
                tactics.get('inward_trap_weight', 50.0),
                tactics.get('inward_trap_min_enemy_length', 5),
                tactics.get('aggressive_space_control_weight', 30.0),
                tactics.get('aggressive_space_turn_threshold', 50),
                tactics.get('predictive_avoidance_weight', 100.0),
                tactics.get('energy_conservation_weight', 15.0),
                tactics.get('adaptive_wall_hugging_weight', 25.0)
            ])
            # Emergency wall escape section (7 params)
            emergency = config.get('emergency_wall_escape', {})
            vector.extend([
                emergency.get('min_distance', 2),
                emergency.get('max_distance', 4),
                emergency.get('turn_bonus', 150.0),
                emergency.get('close_bonus', 200.0),
                emergency.get('away_penalty', 100.0),
                emergency.get('close_threshold', 2),
                emergency.get('coord_tolerance', 3)
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
        print(f"  - Benchmark rounds: {self.benchmark_rounds} (averaging for validation)")
        print(f"  - Test algorithms: {', '.join(self.test_algorithms)}")
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
            baseline_wr, _ = self.run_benchmark(50)  # Ignore death analysis for baseline
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
            
            # Initialize variables that may not be set in all code paths
            win_rate_std = 0.0
            death_analysis = {}
            
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
                
                # Save candidate and reload config (faster than rebuilding)
                self.save_config(candidate_config)
                
                # Try to reload via endpoint first, fall back to rebuild if needed
                reload_success = self.reload_config_via_endpoint()
                if not reload_success:
                    # Fallback to rebuild if endpoint reload fails
                    rebuild_success = self.rebuild_snake()
                    if not rebuild_success:
                        print("❌ Build failed, skipping iteration\n")
                        self.save_config(current_config)  # Restore previous config
                        continue
                
                # Test candidate with multiple rounds for validation
                stats = self.run_benchmark_rounds(self.games_per_iteration, rounds=self.benchmark_rounds)
                win_rate = stats['mean']  # Use average win rate
                win_rate_std = stats['std']
                death_analysis = stats.get('death_analysis', {})
                self.state['total_games'] += self.games_per_iteration * self.benchmark_rounds
            
            duration = time.time() - start_time
            
            # Log iteration with death analysis
            iteration_data = {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'win_rate': win_rate,
                'win_rate_std': win_rate_std,
                'best_win_rate': self.state['best_win_rate'],
                'improvement': win_rate - self.state['best_win_rate'],
                'games': self.games_per_iteration,
                'benchmark_rounds': self.benchmark_rounds,
                'duration_seconds': int(duration),
                'config': candidate_config,
                'death_analysis': death_analysis
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
                
                # Save the improved config to main config.yaml
                self.save_config(candidate_config)
                print(f"   💾 Saved improved config to config.yaml")
                
                # Commit improvement to git
                print(f"   📝 Committing improvement to git...")
                self.git_commit_improvement(win_rate, improvement)
            else:
                print(f"\n⚪ No improvement: {win_rate:.1%} (best: {self.state['best_win_rate']:.1%})")
                # Restore best config
                self.save_config(current_config)
                # Use reload endpoint for faster config restoration
                reload_success = self.reload_config_via_endpoint()
                if not reload_success:
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
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                       help='Battlesnake server URL for config reload endpoint (default: http://localhost:8000)')
    parser.add_argument('--benchmark-rounds', type=int, default=3,
                       help='Number of benchmark rounds to average for validation (default: 3). Higher = more reliable but slower.')
    parser.add_argument('--test-algorithms', type=str, nargs='+', default=['hybrid'],
                       help='Algorithms to test: hybrid, greedy, lookahead, mcts (default: hybrid)')
    
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
        parallel_configs=args.parallel_configs,
        server_url=args.server_url,
        benchmark_rounds=args.benchmark_rounds,
        test_algorithms=args.test_algorithms
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
