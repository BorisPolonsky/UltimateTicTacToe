#!/usr/bin/env python3
"""
Concrete training algorithm implementations for Ultimate Tic-Tac-Toe.
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import Dict, Any, Optional, Callable, List, Tuple
from .env import UltimateTicTacToeEnv
from .abstract_agent import Agent
from .abstract_encoders import StateEncoder
from .abstract_trainer import TrainingAlgorithm


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNTrainingAlgorithm(TrainingAlgorithm):
    """DQN training algorithm."""
    
    def __init__(self, 
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 gamma: float = 0.99,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize DQN training algorithm.
        
        Args:
            memory_size: Size of replay memory
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            gamma: Discount factor
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
        """
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.step_count = 0
    
    def train_episode(self, env: UltimateTicTacToeEnv, agent: Agent, 
                     opponent: Agent, encoder: StateEncoder) -> Dict[str, Any]:
        """Train DQN agent for one episode."""
        observation, info = env.reset()
        total_reward = 0
        step_count = 0
        terminated = False
        losses = []
        
        while not terminated:
            # Determine current agent (alternate between agent and opponent)
            current_agent = agent if step_count % 2 == 0 else opponent
            
            # Encode current state
            current_state = encoder.encode(observation, info)
            
            # Get action
            action = current_agent.get_action(observation, info)
            
            if action is None:
                break
            
            # Take step
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            next_state = encoder.encode(next_observation, next_info)
            
            # Store experience if this is the training agent
            if current_agent == agent:
                self.memory.append(Experience(current_state, action, reward, next_state, terminated))
                
                # Train the agent
                loss = self._train_step(agent)
                if loss is not None:
                    losses.append(loss)
                
                # Decay epsilon
                if agent.epsilon > self.epsilon_min:
                    agent.epsilon *= self.epsilon_decay
            
            # Update state
            observation, info = next_observation, next_info
            total_reward += reward
            step_count += 1
        
        return {
            'total_reward': total_reward,
            'step_count': step_count,
            'winner': info['winner'],
            'avg_loss': np.mean(losses) if losses else 0.0,
            'epsilon': getattr(agent, 'epsilon', 0.0)
        }
    
    def _train_step(self, agent) -> Optional[float]:
        """Train the agent on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(agent.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(agent.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(agent.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(agent.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(agent.device)
        
        # Current Q-values
        current_q_values = agent.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = agent.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backward pass
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
        
        return loss.item()


class Trainer:
    """Flexible trainer for Ultimate Tic-Tac-Toe agents."""
    
    def __init__(self, 
                 env: UltimateTicTacToeEnv,
                 agent: Agent,
                 opponent: Agent,
                 encoder: StateEncoder,
                 training_algorithm: TrainingAlgorithm,
                 evaluation_interval: int = 100,
                 save_interval: int = 1000,
                 model_save_path: str = "models/",
                 log_interval: int = 10):
        """
        Initialize trainer.
        
        Args:
            env: Environment instance
            agent: Agent to train
            opponent: Opponent agent
            encoder: State encoder
            training_algorithm: Training algorithm
            evaluation_interval: How often to evaluate
            save_interval: How often to save model
            model_save_path: Path to save models
            log_interval: How often to log progress
        """
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.encoder = encoder
        self.training_algorithm = training_algorithm
        
        # Training parameters
        self.evaluation_interval = evaluation_interval
        self.save_interval = save_interval
        self.model_save_path = model_save_path
        self.log_interval = log_interval
        
        # Training state
        self.episode_count = 0
        self.training_history = []
        self.evaluation_history = []
        
        # Create model save directory
        import os
        os.makedirs(model_save_path, exist_ok=True)
    
    def train(self, num_episodes: int, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            callbacks: List of callback functions to call after each episode
            
        Returns:
            Training summary
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Agent: {type(self.agent).__name__}")
        print(f"Opponent: {type(self.opponent).__name__}")
        print(f"Encoder: {type(self.encoder).__name__}")
        print(f"Training Algorithm: {type(self.training_algorithm).__name__}")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Train one episode
            metrics = self.training_algorithm.train_episode(
                self.env, self.agent, self.opponent, self.encoder
            )
            
            self.episode_count += 1
            self.training_history.append(metrics)
            
            # Log progress
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1, metrics)
            
            # Evaluate
            if (episode + 1) % self.evaluation_interval == 0:
                eval_metrics = self._evaluate()
                self.evaluation_history.append(eval_metrics)
                self._log_evaluation(episode + 1, eval_metrics)
            
            # Save model
            if (episode + 1) % self.save_interval == 0:
                self._save_model(episode + 1)
            
            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, episode, metrics)
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_eval = self._evaluate()
        
        return {
            'episodes_trained': num_episodes,
            'training_time': training_time,
            'final_evaluation': final_eval,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history
        }
    
    def _evaluate(self, num_games: int = 100) -> Dict[str, Any]:
        """Evaluate the trained agent."""
        print(f"Evaluating agent over {num_games} games...")
        
        # Create evaluation environment
        eval_env = UltimateTicTacToeEnv()
        
        # Set agent to evaluation mode
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(False)
        
        wins = 0
        losses = 0
        draws = 0
        total_rewards = []
        
        for game in range(num_games):
            observation, info = eval_env.reset()
            total_reward = 0
            terminated = False
            step_count = 0
            
            while not terminated:
                # Determine current agent (alternate between agent and opponent)
                current_agent = self.agent if step_count % 2 == 0 else self.opponent
                
                # Get action
                action = current_agent.get_action(observation, info)
                
                if action is None:
                    break
                
                # Take step
                observation, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                step_count += 1
            
            # Record result
            if info['winner'] == 1:  # Player 1 wins
                wins += 1
            elif info['winner'] == 2:  # Player 2 wins
                losses += 1
            else:
                draws += 1
            
            total_rewards.append(total_reward)
        
        eval_env.close()
        
        # Restore training mode
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(True)
        
        return {
            'win_rate': wins / num_games,
            'loss_rate': losses / num_games,
            'draw_rate': draws / num_games,
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards)
        }
    
    def _log_progress(self, episode: int, metrics: Dict[str, Any]):
        """Log training progress."""
        print(f"Episode {episode:4d} | "
              f"Reward: {metrics['total_reward']:6.2f} | "
              f"Steps: {metrics['step_count']:3d} | "
              f"Winner: {metrics['winner']} | "
              f"Loss: {metrics['avg_loss']:6.4f} | "
              f"Epsilon: {metrics['epsilon']:5.3f}")
    
    def _log_evaluation(self, episode: int, metrics: Dict[str, Any]):
        """Log evaluation results."""
        print(f"Evaluation at episode {episode}:")
        print(f"  Win Rate: {metrics['win_rate']:.3f}")
        print(f"  Loss Rate: {metrics['loss_rate']:.3f}")
        print(f"  Draw Rate: {metrics['draw_rate']:.3f}")
        print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
        print("-" * 40)
    
    def _save_model(self, episode: int):
        """Save the trained model."""
        if hasattr(self.agent, 'save_model'):
            filename = f"{self.model_save_path}/model_episode_{episode}.pth"
            self.agent.save_model(filename)
            print(f"Model saved to {filename}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress."""
        if not self.training_history:
            return {}
        
        recent_metrics = self.training_history[-100:]  # Last 100 episodes
        
        return {
            'total_episodes': self.episode_count,
            'avg_reward': np.mean([m['total_reward'] for m in recent_metrics]),
            'avg_steps': np.mean([m['step_count'] for m in recent_metrics]),
            'avg_loss': np.mean([m['avg_loss'] for m in recent_metrics]),
            'current_epsilon': recent_metrics[-1]['epsilon'] if recent_metrics else 0.0
        } 