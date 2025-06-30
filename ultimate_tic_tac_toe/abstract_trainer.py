#!/usr/bin/env python3
"""
Abstract training algorithms for Ultimate Tic-Tac-Toe.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
from .env import UltimateTicTacToeEnv
from .abstract_agent import Agent

if TYPE_CHECKING:
    from .dqn_trainer import MultiPlaneEncoder


class TrainingAlgorithm(ABC):
    """Abstract base class for training algorithms."""
    
    @abstractmethod
    def train_episode(self, env: UltimateTicTacToeEnv, agent: Agent, 
                     opponent: Agent, encoder: 'MultiPlaneEncoder') -> Dict[str, Any]:
        """
        Train for one episode.
        
        Args:
            env: Environment instance
            agent: Agent to train
            opponent: Opponent agent
            encoder: State encoder
            
        Returns:
            Dictionary with training metrics
        """
        pass 