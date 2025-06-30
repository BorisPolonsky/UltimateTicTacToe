#!/usr/bin/env python3
"""
DQN Agent for Ultimate Tic-Tac-Toe.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Dict, Any, Optional, Tuple, List
from .abstract_agent import Agent
from .abstract_encoders import StateEncoder


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network for Ultimate Tic-Tac-Toe."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int = 81, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Determine if input is 3D (multiplane) or 1D (simple)
        if len(input_shape) == 3:  # Multiplane encoder: (3, 3, 23)
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
        # Input: 3x3, after 3 conv layers with padding=1: 6x6
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
                 encoder: StateEncoder,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
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