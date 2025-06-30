#!/usr/bin/env python3
"""
DQN Agent and Trainer for Ultimate Tic-Tac-Toe.
Combines DQN agent implementation with training algorithms and trainer.
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import Dict, Any, Optional, Callable, List, Tuple
from .env import UltimateTicTacToeEnv
from .abstract_agent import Agent
from .abstract_trainer import TrainingAlgorithm


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class MultiPlaneEncoder:
    """
    Multi-plane tensor encoder similar to AlphaGo.
    
    Encodes the game state as a 3x3x31 tensor where:
    - 18 planes for local board states (9 small boards × 2 players)
    - 3 planes for global board state (captured boards)
    - 9 planes for valid moves (one per small board)
    - 1 plane for current player
    
    Total: 3x3x31 tensor
    """
    
    def __init__(self):
        self.num_planes = 31  # 18 local + 3 global + 9 valid + 1 player
        self.input_shape = (3, 3, self.num_planes)
    
    def encode(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Encode the current state into a 3x3x31 tensor.
        
        Planes 0-17: Local Board States (9 small boards × 2 planes each)
        - Planes 0-8: Player X's moves in each small board (binary: 0/1)
        - Planes 9-17: Player O's moves in each small board (binary: 0/1)
        
        Planes 18-20: Global Board State (3 planes)
        - Plane 18: Small boards captured by X (binary: 0/1)
        - Plane 19: Small boards captured by O (binary: 0/1)
        - Plane 20: Draw/stalemate small boards (binary: 0/1)
        
        Planes 21-29: Valid Moves (9 planes)
        - Planes 21-29: For each small board, a 3x3 binary matrix indicating valid moves
        
        Plane 30: Current Player (1 plane)
        - Plane 30: Whose turn it is (binary: 0 = X, 1 = O)
        """
        # Initialize tensor
        tensor = np.zeros(self.input_shape, dtype=np.float32)
        
        # Planes 0-17: Local board states
        for block_row in range(3):
            for block_col in range(3):
                block_idx = block_row * 3 + block_col
                
                # Extract the 3x3 small board
                start_row = block_row * 3
                start_col = block_col * 3
                small_board = observation[start_row:start_row+3, start_col:start_col+3]
                
                # Plane for Player X's moves (planes 0-8) - binary encoding
                tensor[block_row, block_col, block_idx] = 1.0 if np.any(small_board == 1) else 0.0
                
                # Plane for Player O's moves (planes 9-17) - binary encoding
                tensor[block_row, block_col, block_idx + 9] = 1.0 if np.any(small_board == 2) else 0.0
        
        # Planes 18-20: Global board state
        block_status = info['block_status']
        
        # Plane 18: Small boards captured by X
        tensor[:, :, 18] = (block_status == 1).astype(np.float32)
        
        # Plane 19: Small boards captured by O
        tensor[:, :, 19] = (block_status == 2).astype(np.float32)
        
        # Plane 20: Draw/stalemate small boards
        tensor[:, :, 20] = (block_status == 3).astype(np.float32)
        
        # Planes 21-29: Valid moves (one plane per small board)
        valid_moves = info['valid_moves']
        next_block = info.get('next_block', None)
        
        for block_row in range(3):
            for block_col in range(3):
                plane_idx = 21 + block_row * 3 + block_col
                
                # Check if this small board is active (next_block is specified)
                if next_block is not None and next_block != (None, None):
                    # If a specific block is active, only that block's plane has valid moves
                    if (block_row, block_col) == next_block:
                        # Fill this plane with valid moves for the active block
                        for slot_row in range(3):
                            for slot_col in range(3):
                                # Calculate the action index for this position
                                action = self._encode_action(block_row, block_col, slot_row, slot_col)
                                if valid_moves[action] == 1:
                                    tensor[slot_row, slot_col, plane_idx] = 1.0
                else:
                    # If no specific block is active, all valid moves are distributed
                    # This happens when the target block is already won/drawn
                    for slot_row in range(3):
                        for slot_col in range(3):
                            # Calculate the action index for this position
                            action = self._encode_action(block_row, block_col, slot_row, slot_col)
                            if valid_moves[action] == 1:
                                tensor[slot_row, slot_col, plane_idx] = 1.0
        
        # Plane 30: Current player (0 for X, 1 for O)
        current_player = info['next_player']
        if current_player == 1:  # Player X
            tensor[:, :, 30] = 0.0
        elif current_player == 2:  # Player O
            tensor[:, :, 30] = 1.0
        
        return tensor
    
    def _encode_action(self, block_row: int, block_col: int, slot_row: int, slot_col: int) -> int:
        """Encode (block_row, block_col, slot_row, slot_col) to action integer.
        
        Returns action_index = row * 9 + col, where (row, col) is the 9x9 board position.
        
        Relationship between coordinates:
        - (block_row, block_col): Which 3x3 block (0-2, 0-2)
        - (slot_row, slot_col): Position within that 3x3 block (0-2, 0-2)
        - (row, col): Direct position on the 9x9 board (0-8, 0-8)
        
        Conversion: row = block_row * 3 + slot_row, col = block_col * 3 + slot_col
        """
        # Convert block and slot coordinates to board position
        row = block_row * 3 + slot_row
        col = block_col * 3 + slot_col
        
        # Convert board position to action index
        return row * 9 + col
    
    def _decode_action(self, action: int) -> Tuple[int, int, int, int]:
        """Decode action integer to (block_row, block_col, slot_row, slot_col).
        
        Action index is mapped as: action_index = row * 9 + col, where (row, col) is the 9x9 board position.
        
        Relationship between coordinates:
        - (row, col): Direct position on the 9x9 board (0-8, 0-8)
        - (block_row, block_col): Which 3x3 block (0-2, 0-2)
        - (slot_row, slot_col): Position within that 3x3 block (0-2, 0-2)
        
        Conversion: row = block_row * 3 + slot_row, col = block_col * 3 + slot_col
        """
        # Convert action index to board position (row, col)
        row = action // 9
        col = action % 9
        
        # Convert board position to block and slot coordinates
        block_row = row // 3
        block_col = col // 3
        slot_row = row % 3
        slot_col = col % 3
        
        return block_row, block_col, slot_row, slot_col
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoded input tensor."""
        return self.input_shape


class DQNNetwork(nn.Module):
    """Deep Q-Network for Ultimate Tic-Tac-Toe."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int = 81, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Determine if input is 3D (multiplane) or 1D (simple)
        if len(input_shape) == 3:  # Multiplane encoder: (3, 3, 31)
            self._build_conv_network(hidden_size)
        else:  # Simple encoder: (81,)
            self._build_fc_network(hidden_size)
    
    def _build_conv_network(self, hidden_size: int):
        """Build convolutional network for multiplane input."""
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_shape[2], 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, padding=1)
        
        # Calculate flattened size after conv layers
        # Input: 3x3, after conv1 (2x2, padding=1): 4x4
        # After conv2 (2x2, padding=1): 5x5
        # After conv3 (2x2, padding=1): 6x6
        conv_output_size = 256 * 6 * 6  # After 3 conv layers on 3x3 input
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.num_actions)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
    
    def _build_fc_network(self, hidden_size: int):
        """Build fully connected network for simple input."""
        input_size = self.input_shape[0]
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if len(self.input_shape) == 3:  # Multiplane input
            # Reshape from (batch, 3, 3, channels) to (batch, channels, 3, 3)
            x = x.permute(0, 3, 1, 2)
            
            # Convolutional layers
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            
            # Flatten
            x = x.reshape(x.size(0), -1)
        else:  # Simple input
            x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class DQNAgent(Agent):
    """Deep Q-Network agent for Ultimate Tic-Tac-Toe."""
    
    def __init__(self, 
                 env,
                 encoder: MultiPlaneEncoder,
                 learning_rate: float = 1e-5,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 device: str = 'cpu'):
        """
        Initialize DQN Agent.
        
        Args:
            env: Environment instance
            encoder: State encoder
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__(env)
        
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Networks
        input_shape = encoder.get_input_shape()
        self.q_network = DQNNetwork(input_shape).to(self.device)
        self.target_network = DQNNetwork(input_shape).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # Training state
        self.training_enabled = True
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Optional[int]:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            observation: Current board state
            info: Additional information
            
        Returns:
            Action integer or None if no valid actions
        """
        valid_actions = info.get('valid_moves', self.env.get_valid_actions())
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            return None
        
        # Epsilon-greedy action selection
        if self.training_enabled and random.random() < self.epsilon:
            # Random action
            return np.random.choice(valid_action_indices)
        else:
            # Greedy action
            return self._get_greedy_action(observation, info, valid_action_indices)
    
    def _get_greedy_action(self, observation: np.ndarray, info: Dict[str, Any], valid_actions: np.ndarray) -> int:
        """Get the best action according to the Q-network."""
        # Encode state
        state = self.encoder.encode(observation, info)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()
        
        # Mask invalid actions with large negative values
        q_values = q_values.cpu().numpy()
        q_values[~np.isin(np.arange(81), valid_actions)] = -1e6
        
        # Return best action
        return np.argmax(q_values)
    
    def set_training_mode(self, training: bool):
        """Enable or disable training mode."""
        self.training_enabled = training
        if not training:
            self.epsilon = 0.0  # No exploration during evaluation
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
    
    def reset(self):
        """Reset the agent's internal state."""
        pass


class DQNTrainingAlgorithm(TrainingAlgorithm):
    """DQN training algorithm."""
    
    def __init__(self, 
                 memory_size: int = 50000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000,
                 gamma: float = 0.99,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.1):
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
                     opponent: Agent, encoder: MultiPlaneEncoder) -> Dict[str, Any]:
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
            
            # Update state
            observation, info = next_observation, next_info
            total_reward += reward
            step_count += 1
        
        # Decay epsilon per episode (not per step)
        if agent.epsilon > self.epsilon_min:
            agent.epsilon *= self.epsilon_decay
        
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
        
        # Gradient clipping to prevent divergence
        torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=1.0)
        
        agent.optimizer.step()
        
        # Update learning rate
        agent.scheduler.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
        
        return loss.item()


class DQNTrainer:
    """DQN trainer for Ultimate Tic-Tac-Toe agents."""
    
    def __init__(self, 
                 env: UltimateTicTacToeEnv,
                 agent: Agent,
                 opponent: Agent,
                 encoder: MultiPlaneEncoder,
                 training_algorithm: TrainingAlgorithm,
                 evaluation_interval: int = 100,
                 save_interval: int = 1000,
                 model_save_path: str = "models/",
                 log_interval: int = 10):
        """
        Initialize DQN trainer.
        
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