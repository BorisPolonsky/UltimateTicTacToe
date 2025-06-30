#!/usr/bin/env python3
"""
Concrete state encoder implementations for Ultimate Tic-Tac-Toe.
Provides flexible encoding schemes for different training algorithms.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .abstract_encoders import StateEncoder


class MultiPlaneEncoder(StateEncoder):
    """
    Multi-plane tensor encoder similar to AlphaGo.
    
    Encodes the game state as a 3x3x23 tensor where:
    - 18 planes for local board states (9 small boards × 2 players)
    - 3 planes for global board state (captured boards)
    - 1 plane for valid moves
    - 1 plane for current player
    
    Total: 3x3x23 tensor
    """
    
    def __init__(self):
        self.num_planes = 23  # 18 local + 3 global + 1 valid + 1 player
        self.input_shape = (3, 3, self.num_planes)
    
    def encode(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Encode the current state into a 3x3x23 tensor.
        
        Planes 0-17: Local board states (9 small boards × 2 planes each)
        - Planes 0-8: Player X's moves in each small board (density)
        - Planes 9-17: Player O's moves in each small board (density)
        
        Planes 18-20: Global board state (3 planes)
        - Plane 18: Small boards captured by X
        - Plane 19: Small boards captured by O  
        - Plane 20: Draw/stalemate small boards
        
        Plane 21: Valid moves (1 plane)
        Plane 22: Current player (1 plane)
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
                
                # Count moves for each player in this small board
                x_count = np.sum(small_board == 1)
                o_count = np.sum(small_board == 2)
                
                # Assign density values to the corresponding planes
                # Plane for Player X's moves (planes 0-8)
                tensor[block_row, block_col, block_idx] = x_count / 9.0
                
                # Plane for Player O's moves (planes 9-17)
                tensor[block_row, block_col, block_idx + 9] = o_count / 9.0
        
        # Planes 18-20: Global board state
        block_status = info['block_status']
        
        # Plane 18: Small boards captured by X
        tensor[:, :, 18] = (block_status == 1).astype(np.float32)
        
        # Plane 19: Small boards captured by O
        tensor[:, :, 19] = (block_status == 2).astype(np.float32)
        
        # Plane 20: Draw/stalemate small boards
        tensor[:, :, 20] = (block_status == 3).astype(np.float32)
        
        # Plane 21: Valid moves (density per block)
        valid_moves = info['valid_moves']
        for block_row in range(3):
            for block_col in range(3):
                # Count valid moves in this block
                valid_count = 0
                for action in range(81):
                    if valid_moves[action] == 1:
                        act_block_row, act_block_col, _, _ = self._decode_action(action)
                        if act_block_row == block_row and act_block_col == block_col:
                            valid_count += 1
                tensor[block_row, block_col, 21] = valid_count / 9.0
        
        # Plane 22: Current player (0 for X, 1 for O)
        current_player = info['next_player']
        if current_player == 1:  # Player X
            tensor[:, :, 22] = 0.0
        elif current_player == 2:  # Player O
            tensor[:, :, 22] = 1.0
        
        return tensor
    
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


class SimpleEncoder(StateEncoder):
    """
    Simple encoder that flattens the 9x9 board into a 1D vector.
    Useful for simple neural networks or as a baseline.
    """
    
    def __init__(self):
        self.input_shape = (81,)
    
    def encode(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Encode the current state into a flattened 1D vector.
        
        Args:
            observation: Current board state (9x9 numpy array)
            info: Additional information from environment
            
        Returns:
            Flattened state vector of shape (81,)
        """
        # Flatten the 9x9 board
        return observation.flatten().astype(np.float32)
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoded input tensor."""
        return self.input_shape


class EncoderFactory:
    """Factory class for creating different types of encoders."""
    
    @staticmethod
    def create_encoder(encoder_type: str) -> StateEncoder:
        """
        Create an encoder of the specified type.
        
        Args:
            encoder_type: Type of encoder ('multiplane', 'simple')
            
        Returns:
            StateEncoder instance
        """
        if encoder_type.lower() == 'multiplane':
            return MultiPlaneEncoder()
        elif encoder_type.lower() == 'simple':
            return SimpleEncoder()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @staticmethod
    def list_available_encoders() -> list:
        """List all available encoder types."""
        return ['multiplane', 'simple'] 