#!/usr/bin/env python3
"""
Reinforcement Learning Trainer for Battlesnake
Uses PPO (Proximal Policy Optimization) with self-play
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import subprocess
import json
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
import random

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for Battlesnake
    Input: Game state (board, snake, food, health)
    Output: Action probabilities + state value
    """
    def __init__(self, state_dim=165, action_dim=4, hidden_dim=512):
        super(PolicyNetwork, self).__init__()
        
        # State encoding layers
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        x = self.encoder(state)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            action_logits, value = self.forward(state)
            probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            return action, probs, value


class ExperienceBuffer:
    """Replay buffer for storing experiences"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, log_probs, values = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            torch.stack(log_probs),
            torch.stack(values)
        )
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO training algorithm"""
    def __init__(self, policy_net, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, epochs=10, batch_size=64):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO policy update"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.epochs):
            # Forward pass
            action_logits, values = self.policy_net(states)
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus (encourages exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.epochs


class RLTrainingSystem:
    """Main RL training orchestrator"""
    def __init__(self, episodes=10000, max_steps=500):
        self.episodes = episodes
        self.max_steps = max_steps
        
        self.results_dir = Path("rl_training_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.checkpoint_file = self.results_dir / "checkpoint.pt"
        self.best_model_file = self.results_dir / "best_model.pt"
        self.log_file = self.results_dir / "training.log"
        
        # Initialize network
        self.policy_net = PolicyNetwork().to(device)
        self.trainer = PPOTrainer(self.policy_net)
        self.buffer = ExperienceBuffer(capacity=10000)
        
        # Training state
        self.episode = 0
        self.best_reward = -float('inf')
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def handle_shutdown(self, signum, frame):
        print("\n⚠ Shutdown signal received, saving checkpoint...")
        self.save_checkpoint()
        print("✓ Checkpoint saved")
        self.running = False
        sys.exit(0)
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'best_reward': self.best_reward,
            'policy_state': self.policy_net.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.checkpoint_file)
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            checkpoint = torch.load(self.checkpoint_file, map_location=device)
            self.episode = checkpoint['episode']
            self.best_reward = checkpoint['best_reward']
            self.policy_net.load_state_dict(checkpoint['policy_state'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"✓ Resumed from episode {self.episode}, best reward: {self.best_reward:.2f}")
    
    def encode_game_state(self, state_dict):
        """Convert game state to neural network input"""
        # This is a simplified version - you'd need actual game state from battlesnake
        # For now, return random state for demonstration
        return torch.randn(1, 165).to(device)
    
    def simulate_episode(self):
        """Simulate one episode (game)"""
        # This would run actual battlesnake game
        # For now, simplified simulation
        
        total_reward = 0
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        state = torch.randn(1, 165).to(device)  # Initial state
        
        for step in range(self.max_steps):
            # Get action from policy
            action, probs, value = self.policy_net.get_action(state)
            log_prob = torch.log(probs[0, action])
            
            # Simulate environment step
            # In real implementation, this would interact with battlesnake game
            reward = random.uniform(-1, 1)  # Placeholder
            next_state = torch.randn(1, 165).to(device)
            done = random.random() < 0.01  # Small chance of episode end
            
            # Store experience
            states.append(state[0])
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value[0])
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return total_reward, states, actions, rewards, log_probs, values
    
    def train(self):
        """Main training loop"""
        print("="*70)
        print("🎮 REINFORCEMENT LEARNING TRAINING")
        print("="*70)
        print(f"Episodes: {self.episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        print(f"Starting episode: {self.episode}")
        print(f"Best reward: {self.best_reward:.2f}")
        print("="*70)
        print()
        
        while self.running and self.episode < self.episodes:
            self.episode += 1
            
            # Run episode
            start_time = time.time()
            total_reward, states, actions, rewards, log_probs, values = self.simulate_episode()
            
            # Convert to tensors
            states_tensor = torch.stack(states)
            actions_tensor = torch.tensor(actions, device=device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            log_probs_tensor = torch.stack(log_probs)
            values_tensor = torch.stack(values).squeeze()
            
            # Compute returns and advantages
            next_values = torch.cat([values_tensor[1:], torch.zeros(1, device=device)])
            dones = torch.zeros_like(rewards_tensor)
            dones[-1] = 1.0
            
            advantages, returns = self.trainer.compute_gae(
                rewards_tensor, values_tensor, next_values, dones
            )
            
            # Update policy
            loss = self.trainer.update(
                states_tensor, actions_tensor, log_probs_tensor, 
                returns, advantages
            )
            
            duration = time.time() - start_time
            
            # Log progress
            log_msg = f"Episode {self.episode}/{self.episodes} | " \
                     f"Reward: {total_reward:.2f} | " \
                     f"Loss: {loss:.4f} | " \
                     f"Time: {duration:.1f}s"
            print(log_msg)
            
            with open(self.log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            # Save best model
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                torch.save({
                    'episode': self.episode,
                    'reward': total_reward,
                    'policy_state': self.policy_net.state_dict()
                }, self.best_model_file)
                print(f"  ✅ New best reward: {total_reward:.2f}")
            
            # Periodic checkpoint
            if self.episode % 100 == 0:
                self.save_checkpoint()
                print(f"  💾 Checkpoint saved at episode {self.episode}")
        
        print("\n" + "="*70)
        print("🏁 TRAINING COMPLETE")
        print("="*70)
        print(f"Total episodes: {self.episode}")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Best model: {self.best_model_file}")
        print("="*70)
        
        self.save_checkpoint()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Training for Battlesnake')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes (default: 10000)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode (default: 500)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    
    args = parser.parse_args()
    
    trainer = RLTrainingSystem(
        episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
