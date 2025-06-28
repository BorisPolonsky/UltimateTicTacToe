import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import random


class Board:
    """
    Ultimate Tic-Tac-Toe board represented as a 9x9 numpy array.
    0 = unoccupied, 1 = player 1, 2 = player 2, 3 = draw
    """
    
    def __init__(self, sovereignty_upon_draw: str = "none"):
        """
        Initialize the board.
        
        Args:
            sovereignty_upon_draw: "none" or "both" for rule variants
        """
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.next_player = 1  # Player 1 always starts
        self.next_block = None  # (row, col) of the next block to play in
        self.game_over = False
        self.winner = None
        self.sovereignty_upon_draw = sovereignty_upon_draw
        
        # Track completed blocks (0=empty, 1=player1, 2=player2, 3=draw)
        self.block_status = np.zeros((3, 3), dtype=np.int8)
        self.num_filled_blocks = 0
    
    def get_block(self, block_row: int, block_col: int) -> np.ndarray:
        """Get a 3x3 block from the 9x9 board."""
        start_row = block_row * 3
        start_col = block_col * 3
        return self.board[start_row:start_row+3, start_col:start_col+3]
    
    def set_block(self, block_row: int, block_col: int, block_data: np.ndarray):
        """Set a 3x3 block in the 9x9 board."""
        start_row = block_row * 3
        start_col = block_col * 3
        self.board[start_row:start_row+3, start_col:start_col+3] = block_data
    
    def is_valid_move(self, block_row: int, block_col: int, slot_row: int, slot_col: int) -> bool:
        """Check if a move is valid."""
        if self.game_over:
            return False
        
        # Check if the block is valid
        if self.next_block is not None:
            if (block_row, block_col) != self.next_block:
                return False
        
        # Check if the block is completed
        if self.block_status[block_row, block_col] != 0:
            return False
        
        # Check if the slot is empty
        start_row = block_row * 3 + slot_row
        start_col = block_col * 3 + slot_col
        return self.board[start_row, start_col] == 0
    
    def make_move(self, block_row: int, block_col: int, slot_row: int, slot_col: int) -> bool:
        """
        Make a move and return True if the game is over.
        
        Args:
            block_row, block_col: Block coordinates (0-2)
            slot_row, slot_col: Slot coordinates within block (0-2)
        
        Returns:
            True if game is over, False otherwise
        """
        if not self.is_valid_move(block_row, block_col, slot_row, slot_col):
            raise ValueError(f"Invalid move: ({block_row}, {block_col}, {slot_row}, {slot_col})")
        
        # Make the move
        start_row = block_row * 3 + slot_row
        start_col = block_col * 3 + slot_col
        self.board[start_row, start_col] = self.next_player
        
        # Check if the block is completed
        block = self.get_block(block_row, block_col)
        block_winner = self._check_block_winner(block)
        
        if block_winner != 0:
            self.block_status[block_row, block_col] = block_winner
            self.num_filled_blocks += 1
            
            # Check if the game is won
            if self._check_game_winner():
                self.game_over = True
                self.winner = self.next_player
                return True
            
            # Check if it's a draw
            if self.num_filled_blocks == 9:
                self.game_over = True
                self.winner = 0  # Draw
                return True
        
        # Determine next block
        target_block = (slot_row, slot_col)
        if self.block_status[target_block[0], target_block[1]] == 0:
            self.next_block = target_block
        else:
            self.next_block = None
        
        # Switch players
        self.next_player = 3 - self.next_player  # 1 -> 2, 2 -> 1
        
        return False
    
    def _check_block_winner(self, block: np.ndarray) -> int:
        """Check if a 3x3 block has a winner. Returns 0 (no winner), 1 (player1), 2 (player2), or 3 (draw)."""
        # Check rows
        for row in range(3):
            if block[row, 0] != 0 and block[row, 0] == block[row, 1] == block[row, 2]:
                return block[row, 0]
        
        # Check columns
        for col in range(3):
            if block[0, col] != 0 and block[0, col] == block[1, col] == block[2, col]:
                return block[0, col]
        
        # Check diagonals
        if block[0, 0] != 0 and block[0, 0] == block[1, 1] == block[2, 2]:
            return block[0, 0]
        
        if block[0, 2] != 0 and block[0, 2] == block[1, 1] == block[2, 0]:
            return block[0, 2]
        
        # Check for draw
        if np.all(block != 0):
            return 3  # Draw
        
        return 0  # No winner yet
    
    def _check_game_winner(self) -> bool:
        """Check if the game has a winner."""
        # Apply draw rule if needed
        status = self.block_status.copy()
        if self.sovereignty_upon_draw == "both":
            # Count draws as wins for the current player
            draw_mask = (status == 3)
            status[draw_mask] = self.next_player
        
        # Check rows
        for row in range(3):
            if status[row, 0] != 0 and status[row, 0] == status[row, 1] == status[row, 2]:
                return True
        
        # Check columns
        for col in range(3):
            if status[0, col] != 0 and status[0, col] == status[1, col] == status[2, col]:
                return True
        
        # Check diagonals
        if status[0, 0] != 0 and status[0, 0] == status[1, 1] == status[2, 2]:
            return True
        
        if status[0, 2] != 0 and status[0, 2] == status[1, 1] == status[2, 0]:
            return True
        
        return False
    
    def get_valid_moves(self) -> List[Tuple[int, int, int, int]]:
        """Get all valid moves as (block_row, block_col, slot_row, slot_col)."""
        if self.game_over:
            return []
        
        valid_moves = []
        
        if self.next_block is None:
            # Can play in any uncompleted block
            for block_row in range(3):
                for block_col in range(3):
                    if self.block_status[block_row, block_col] == 0:
                        block = self.get_block(block_row, block_col)
                        for slot_row in range(3):
                            for slot_col in range(3):
                                if block[slot_row, slot_col] == 0:
                                    valid_moves.append((block_row, block_col, slot_row, slot_col))
        else:
            # Must play in the specified block
            block_row, block_col = self.next_block
            block = self.get_block(block_row, block_col)
            for slot_row in range(3):
                for slot_col in range(3):
                    if block[slot_row, slot_col] == 0:
                        valid_moves.append((block_row, block_col, slot_row, slot_col))
        
        return valid_moves
    
    def get_random_move(self) -> Optional[Tuple[int, int, int, int]]:
        """Get a random valid move."""
        valid_moves = self.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        return {
            'board': self.board.copy(),
            'block_status': self.block_status.copy(),
            'next_player': self.next_player,
            'next_block': self.next_block,
            'game_over': self.game_over,
            'winner': self.winner,
            'num_filled_blocks': self.num_filled_blocks
        }
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(sovereignty_upon_draw=self.sovereignty_upon_draw)
        new_board.board = self.board.copy()
        new_board.block_status = self.block_status.copy()
        new_board.next_player = self.next_player
        new_board.next_block = self.next_block
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        new_board.num_filled_blocks = self.num_filled_blocks
        return new_board
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = []
        for block_row in range(3):
            for slot_row in range(3):
                row_parts = []
                for block_col in range(3):
                    block = self.get_block(block_row, block_col)
                    row_parts.append(" | ".join(self._player_to_symbol(block[slot_row, :])))
                result.append(" " + "||".join(row_parts) + " ")
            
            if block_row < 2:
                result.append("-" * 33)
        
        return "\n".join(result)
    
    def _player_to_symbol(self, row: np.ndarray) -> List[str]:
        """Convert player numbers to symbols for display."""
        symbols = []
        for cell in row:
            if cell == 0:
                symbols.append(" ")
            elif cell == 1:
                symbols.append("X")
            else:
                symbols.append("O")
        return symbols
    
    def __repr__(self) -> str:
        return f"Board(next_player={self.next_player}, game_over={self.game_over}, winner={self.winner})" 