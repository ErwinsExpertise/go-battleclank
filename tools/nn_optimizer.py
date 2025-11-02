#!/usr/bin/env python3
"""
Neural Network-Based Weight Optimizer for Battlesnake
Uses GPU acceleration with PyTorch to find optimal weight configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class WeightOptimizer(nn.Module):
    """
    Neural network that learns optimal weight configurations
    Input: Current weights + game statistics
    Output: Adjusted weights that maximize win rate
    Enhanced to support ALL config parameters (36 parameters)
    """
    def __init__(self, input_size=36, hidden_size=256):
        super(WeightOptimizer, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, input_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class BattlesnakeOptimizer:
    """Main optimizer that uses NN to find optimal weights"""
    
    def __init__(self, config_path="config.yaml", num_games=50):
        self.config_path = config_path
        self.num_games = num_games
        self.model = WeightOptimizer().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_win_rate = 0.0
        self.best_config = None
        self.history = []
        
    def load_config(self):
        """Load current config.yaml"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config):
        """Save modified config.yaml"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def extract_weights_vector(self, config):
        """Extract weight parameters into a flat vector - ALL config parameters (36 total)"""
        weights = [
            # Weights section (6 params)
            config['weights']['space'],
            config['weights']['head_collision'],
            config['weights']['center_control'],
            config['weights']['wall_penalty'],
            config['weights']['cutoff'],
            config['weights']['food'],
            # Traps section (5 params)
            config['traps']['moderate'],
            config['traps']['severe'],
            config['traps']['critical'],
            config['traps']['food_trap'],
            config['traps']['food_trap_threshold'],
            # Pursuit section (4 params)
            config['pursuit']['distance_2'],
            config['pursuit']['distance_3'],
            config['pursuit']['distance_4'],
            config['pursuit']['distance_5'],
            # Trapping section (3 params)
            config['trapping']['weight'],
            config['trapping']['space_cutoff_threshold'],
            config['trapping']['trapped_ratio'],
            # Food urgency section (3 params)
            config['food_urgency']['critical'],
            config['food_urgency']['low'],
            config['food_urgency']['normal'],
            # Late game section (2 params)
            config['late_game']['caution_multiplier'],
            config['late_game']['turn_threshold'],
            # Hybrid section (6 params - numeric only)
            config['hybrid']['critical_health'],
            config['hybrid']['critical_nearby_enemies'],
            config['hybrid']['critical_space_ratio'],
            config['hybrid']['lookahead_depth'],
            config['hybrid']['mcts_iterations'],
            config['hybrid']['mcts_timeout_ms'],
            # Search section (2 params)
            config['search']['max_astar_nodes'],
            config['search']['max_depth'],
            # Optimization section (5 params)
            config['optimization']['learning_rate'],
            config['optimization']['discount_factor'],
            config['optimization']['exploration_rate'],
            config['optimization']['batch_size'],
            config['optimization']['episodes'],
        ]
        return np.array(weights, dtype=np.float32)
    
    def apply_weights_vector(self, config, weights):
        """Apply weight vector back to config - ALL 36 parameters"""
        # Weights section (6 params: 0-5)
        config['weights']['space'] = float(max(1.0, weights[0]))
        config['weights']['head_collision'] = float(max(100.0, weights[1]))
        config['weights']['center_control'] = float(max(0.5, weights[2]))
        config['weights']['wall_penalty'] = float(max(1.0, weights[3]))
        config['weights']['cutoff'] = float(max(50.0, weights[4]))
        config['weights']['food'] = float(max(0.1, weights[5]))
        # Traps section (5 params: 6-10)
        config['traps']['moderate'] = float(max(100.0, weights[6]))
        config['traps']['severe'] = float(max(200.0, weights[7]))
        config['traps']['critical'] = float(max(300.0, weights[8]))
        config['traps']['food_trap'] = float(max(400.0, weights[9]))
        config['traps']['food_trap_threshold'] = float(max(0.5, min(0.95, weights[10])))
        # Pursuit section (4 params: 11-14)
        config['pursuit']['distance_2'] = float(max(50.0, weights[11]))
        config['pursuit']['distance_3'] = float(max(25.0, weights[12]))
        config['pursuit']['distance_4'] = float(max(10.0, weights[13]))
        config['pursuit']['distance_5'] = float(max(5.0, weights[14]))
        # Trapping section (3 params: 15-17)
        config['trapping']['weight'] = float(max(100.0, weights[15]))
        config['trapping']['space_cutoff_threshold'] = float(max(0.1, min(0.5, weights[16])))
        config['trapping']['trapped_ratio'] = float(max(0.3, min(0.9, weights[17])))
        # Food urgency section (3 params: 18-20)
        config['food_urgency']['critical'] = float(max(1.0, min(3.0, weights[18])))
        config['food_urgency']['low'] = float(max(1.0, min(2.0, weights[19])))
        config['food_urgency']['normal'] = float(max(0.5, min(1.5, weights[20])))
        # Late game section (2 params: 21-22)
        config['late_game']['caution_multiplier'] = float(max(1.0, min(2.0, weights[21])))
        config['late_game']['turn_threshold'] = int(max(50, min(300, weights[22])))
        # Hybrid section (6 params: 23-28)
        config['hybrid']['critical_health'] = int(max(10, min(50, weights[23])))
        config['hybrid']['critical_nearby_enemies'] = int(max(1, min(4, weights[24])))
        config['hybrid']['critical_space_ratio'] = float(max(1.0, min(5.0, weights[25])))
        config['hybrid']['lookahead_depth'] = int(max(1, min(5, weights[26])))
        config['hybrid']['mcts_iterations'] = int(max(50, min(200, weights[27])))
        config['hybrid']['mcts_timeout_ms'] = int(max(100, min(500, weights[28])))
        # Search section (2 params: 29-30)
        config['search']['max_astar_nodes'] = int(max(100, min(1000, weights[29])))
        config['search']['max_depth'] = int(max(50, min(200, weights[30])))
        # Optimization section (5 params: 31-35)
        config['optimization']['learning_rate'] = float(max(0.0001, min(0.1, weights[31])))
        config['optimization']['discount_factor'] = float(max(0.8, min(0.99, weights[32])))
        config['optimization']['exploration_rate'] = float(max(0.01, min(0.5, weights[33])))
        config['optimization']['batch_size'] = int(max(16, min(128, weights[34])))
        config['optimization']['episodes'] = int(max(500, min(5000, weights[35])))
        return config
    
    def run_benchmark(self):
        """Run benchmark and return win rate"""
        print(f"Running {self.num_games} games benchmark...")
        
        # Build Go snake
        build_result = subprocess.run(
            ["go", "build", "-o", "battlesnake"],
            cwd="/home/runner/work/go-battleclank/go-battleclank",
            capture_output=True
        )
        
        if build_result.returncode != 0:
            print(f"Build failed: {build_result.stderr.decode()}")
            return 0.0
        
        # Run benchmark
        result = subprocess.run(
            ["python3", "tools/run_benchmark.py", str(self.num_games)],
            cwd="/home/runner/work/go-battleclank/go-battleclank",
            capture_output=True,
            text=True
        )
        
        # Parse win rate from output
        for line in result.stdout.split('\n'):
            if "Win rate" in line or "Wins:" in line:
                try:
                    # Extract percentage
                    if '%' in line:
                        rate = float(line.split('(')[1].split('%')[0])
                        return rate / 100.0
                    elif '/' in line:
                        wins = int(line.split(':')[1].strip().split()[0])
                        return wins / self.num_games
                except:
                    continue
        
        return 0.0
    
    def train_epoch(self, num_iterations=10):
        """Train for one epoch"""
        print(f"\n{'='*60}")
        print(f"Training Epoch - {num_iterations} iterations")
        print(f"{'='*60}")
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Load current config
            config = self.load_config()
            current_weights = self.extract_weights_vector(config)
            
            # Convert to tensor
            weights_tensor = torch.tensor(current_weights, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get model prediction (weight adjustments)
            self.model.train()
            predicted_adjustments = self.model(weights_tensor)
            
            # Apply adjustments (small steps for stability)
            adjusted_weights = current_weights + predicted_adjustments.cpu().detach().numpy().flatten() * 0.1
            
            # Apply to config and save
            adjusted_config = self.apply_weights_vector(config.copy(), adjusted_weights)
            self.save_config(adjusted_config)
            
            print(f"Testing configuration with adjusted weights...")
            print(f"  Space: {current_weights[0]:.1f} → {adjusted_weights[0]:.1f}")
            print(f"  Pursuit[2]: {current_weights[10]:.1f} → {adjusted_weights[10]:.1f}")
            print(f"  Trap[critical]: {current_weights[8]:.1f} → {adjusted_weights[8]:.1f}")
            
            # Run benchmark
            win_rate = self.run_benchmark()
            
            print(f"Win rate: {win_rate*100:.1f}%")
            
            # Calculate reward (win rate improvement)
            reward = win_rate - self.best_win_rate
            
            # Store in history
            self.history.append({
                'iteration': len(self.history) + 1,
                'win_rate': win_rate,
                'weights': adjusted_weights.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best if improved
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.best_config = adjusted_config.copy()
                print(f"✓ NEW BEST: {win_rate*100:.1f}%")
            
            # Calculate loss (negative reward)
            loss = -reward
            loss_tensor = torch.tensor(loss, dtype=torch.float32).to(device)
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss_tensor.backward()
            self.optimizer.step()
            
            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_iter_{iteration+1}.pt")
        
        print(f"\nEpoch complete. Best win rate: {self.best_win_rate*100:.1f}%")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_win_rate': self.best_win_rate,
            'best_config': self.best_config,
            'history': self.history
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_win_rate = checkpoint['best_win_rate']
        self.best_config = checkpoint['best_config']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {filename}")
        print(f"Best win rate: {self.best_win_rate*100:.1f}%")
    
    def save_results(self, filename="nn_optimization_results.json"):
        """Save optimization results"""
        results = {
            'best_win_rate': self.best_win_rate,
            'best_config': self.best_config,
            'history': self.history,
            'total_games': len(self.history) * self.num_games
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")
        print(f"Total games played: {len(self.history) * self.num_games}")
        print(f"Best win rate achieved: {self.best_win_rate*100:.1f}%")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Neural Network Weight Optimizer for Battlesnake        ║
    ║  GPU-Accelerated Hyperparameter Tuning                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    NUM_GAMES_PER_TEST = 30  # Balance between speed and accuracy
    NUM_ITERATIONS = 20       # Number of weight configurations to test
    
    optimizer = BattlesnakeOptimizer(
        config_path="config.yaml",
        num_games=NUM_GAMES_PER_TEST
    )
    
    # Train
    optimizer.train_epoch(num_iterations=NUM_ITERATIONS)
    
    # Save best config
    if optimizer.best_config:
        optimizer.save_config(optimizer.best_config)
        print(f"\n✓ Best configuration saved to config.yaml")
    
    # Save detailed results
    optimizer.save_results()
    optimizer.save_checkpoint("final_model.pt")
    
    print("\n" + "="*60)
    print(f"Optimization Complete!")
    print(f"Best Win Rate: {optimizer.best_win_rate*100:.1f}%")
    print(f"Total Games: {len(optimizer.history) * NUM_GAMES_PER_TEST}")
    print("="*60)


if __name__ == "__main__":
    main()
