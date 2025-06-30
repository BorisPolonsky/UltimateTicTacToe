#!/usr/bin/env python3
"""
Abstract agent classes for Ultimate Tic-Tac-Toe.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
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