#!/usr/bin/env python3
"""
Abstract state encoders for Ultimate Tic-Tac-Toe.
Provides abstract base classes for different encoding schemes.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class StateEncoder(ABC):
    """Abstract base class for state encoders."""
    
    @abstractmethod
    def encode(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Encode the current state into a tensor representation.
        
        Args:
            observation: Current board state (9x9 numpy array)
            info: Additional information from environment
            
        Returns:
            Encoded state tensor
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoded input tensor."""
        pass 