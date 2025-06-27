#!/usr/bin/env python3
"""
Agent classes for Ultimate Tic-Tac-Toe.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from .env import UltimateTicTacToeEnv


class Agent(ABC):
    """
    Abstract base class for Ultimate Tic-Tac-Toe agents.
    """
    
    def __init__(self, env: UltimateTicTacToeEnv):
        """
        Initialize the agent.
        
        Args:
            env: The environment the agent will interact with
        """
        self.env = env
    
    @abstractmethod
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Optional[int]:
        """
        Get the next action to take.
        
        Args:
            observation: Current board state (9x9 numpy array)
            info: Additional information from env.step()
        
        Returns:
            Action integer (0-80) or None if no valid actions
        """
        pass
    
    def reset(self):
        """Reset the agent's internal state."""
        pass


class RandomAgent(Agent):
    """
    Simple random agent that chooses valid actions randomly.
    """
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Optional[int]:
        """
        Choose a random valid action.
        
        Args:
            observation: Current board state (9x9 numpy array)
            info: Additional information from env.step()
        
        Returns:
            Random valid action integer (0-80) or None if no valid actions
        """
        valid_actions = info.get('valid_moves', self.env.get_valid_actions())
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            return None
        
        return np.random.choice(valid_action_indices)


class HumanAgent(Agent):
    """
    Human agent that gets actions from user input.
    """
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Optional[int]:
        """
        Get action from human input.
        
        Args:
            observation: Current board state (9x9 numpy array)
            info: Additional information from env.step()
        
        Returns:
            Action integer (0-80) or None to quit
        """
        valid_actions = info.get('valid_moves', self.env.get_valid_actions())
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            return None
        
        print(f"\nValid moves: {len(valid_action_indices)}")
        
        while True:
            try:
                # Get input from user
                move_input = input("Enter your move (block_row,block_col,slot_row,slot_col) or 'q' to quit: ").strip()
                
                if move_input.lower() == 'q':
                    return None
                
                # Parse the move
                parts = move_input.split(',')
                if len(parts) != 4:
                    print("Invalid format. Use: block_row,block_col,slot_row,slot_col")
                    continue
                
                block_row, block_col, slot_row, slot_col = map(int, parts)
                
                # Validate coordinates
                if not (0 <= block_row <= 2 and 0 <= block_col <= 2 and 0 <= slot_row <= 2 and 0 <= slot_col <= 2):
                    print("Coordinates must be between 0 and 2")
                    continue
                
                # Convert to action
                action = self.env._encode_action(block_row, block_col, slot_row, slot_col)
                
                # Check if action is valid
                if valid_actions[action] == 0:
                    print("Invalid move! That position is not available.")
                    continue
                
                return action
                
            except ValueError:
                print("Invalid input. Please enter four numbers separated by commas.")
            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return None 