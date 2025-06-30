import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any, Optional
from .board import Board


class UltimateTicTacToeEnv(gym.Env):
    """
    OpenAI Gym environment for Ultimate Tic-Tac-Toe.
    
    Action space: Discrete(81) representing all possible moves
    Observation space: Box(9, 9) representing the board state
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, 
                 sovereignty_upon_draw: str = "none",
                 render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            sovereignty_upon_draw: "none" or "both" for rule variants
            render_mode: "human" or "rgb_array"
        """
        super().__init__()
        
        self.sovereignty_upon_draw = sovereignty_upon_draw
        self.render_mode = render_mode
        
        # Action space: 81 possible moves (9x9 board)
        # Actions are encoded as: action_index = row * 9 + col
        # one_hot(action_index).reshape(9,9) is a mask for the move
        self.action_space = spaces.Discrete(81)
        
        # Observation space: 9x9 board (0=empty, 1=player1, 2=player2)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(9, 9), dtype=np.int8
        )
        
        # Additional observation info
        self.info_space = spaces.Dict({
            'block_status': spaces.Box(low=0, high=3, shape=(3, 3), dtype=np.int8),  # 0=empty, 1=player1, 2=player2, 3=draw
            'next_player': spaces.Discrete(3),  # 0=game_over, 1=player1, 2=player2
            'next_block': spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))),
            'valid_moves': spaces.Box(low=0, high=80, shape=(81,), dtype=np.int8),
            'game_over': spaces.Discrete(2),
            'winner': spaces.Discrete(3)  # 0=no_winner, 1=player1, 2=player2
        })
        
        self.board = None
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = Board(sovereignty_upon_draw=self.sovereignty_upon_draw)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer from 0-80 representing the move
            
        Returns:
            observation: Current board state
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Decode action to move coordinates
        block_row, block_col, slot_row, slot_col = self._decode_action(action)
        
        # Check if move is valid
        if not self.board.is_valid_move(block_row, block_col, slot_row, slot_col):
            # Invalid move - penalize heavily
            observation = self._get_observation()
            info = self._get_info()
            return observation, -100.0, True, False, info
        
        # Make the move
        game_over = self.board.make_move(block_row, block_col, slot_row, slot_col)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Determine if episode is done
        terminated = game_over
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
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
    
    def _get_observation(self) -> np.ndarray:
        """Get the current board observation."""
        return self.board.board.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        valid_moves = self.board.get_valid_moves()
        valid_actions = np.zeros(81, dtype=np.int8)
        
        for move in valid_moves:
            action = self._encode_action(*move)
            valid_actions[action] = 1
        
        return {
            'block_status': self.board.block_status.copy(),
            'next_player': self.board.next_player,
            'next_block': self.board.next_block if self.board.next_block else (0, 0),
            'valid_moves': valid_actions,
            'game_over': int(self.board.game_over),
            'winner': self.board.winner if self.board.winner is not None else 0
        }
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state."""
        if not self.board.game_over:
            # No reward for non-terminal states
            return 0.0
        
        if self.board.winner == 0:
            # Draw
            return 0.0
        elif self.board.winner == 1:
            # Player 1 wins
            return 1.0
        else:
            # Player 2 wins
            return -1.0
    
    def render(self):
        """Render the current state."""
        if self.render_mode == "human":
            print(self.board)
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for visualization."""
        # Create a simple RGB representation
        # This is a basic implementation - you might want to enhance it
        img = np.zeros((270, 270, 3), dtype=np.uint8)
        
        # Draw grid lines
        img[::30, :] = [128, 128, 128]  # Horizontal lines
        img[:, ::30] = [128, 128, 128]  # Vertical lines
        img[::90, :] = [255, 255, 255]  # Thick horizontal lines
        img[:, ::90] = [255, 255, 255]  # Thick vertical lines
        
        # Draw pieces
        for i in range(9):
            for j in range(9):
                if self.board.board[i, j] == 1:  # Player 1
                    # Draw Player 1 in red
                    x, y = j * 30 + 15, i * 30 + 15
                    img[y-10:y+10, x-10:x+10] = [255, 0, 0]
                elif self.board.board[i, j] == 2:  # Player 2
                    # Draw Player 2 in blue
                    x, y = j * 30 + 15, i * 30 + 15
                    img[y-10:y+10, x-10:x+10] = [0, 0, 255]
        
        return img
    
    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions."""
        return self._get_info()['valid_moves']
    
    def get_state(self) -> Dict[str, Any]:
        """Get the full state of the environment."""
        return self.board.get_state()
    
    def set_state(self, state: Dict[str, Any]):
        """Set the state of the environment."""
        self.board = Board(sovereignty_upon_draw=self.sovereignty_upon_draw)
        self.board.board = state['board'].copy()
        self.board.block_status = state['block_status'].copy()
        self.board.next_player = state['next_player']
        self.board.next_block = state['next_block']
        self.board.game_over = state['game_over']
        self.board.winner = state['winner']
        self.board.num_filled_blocks = state['num_filled_blocks']
    
    def close(self):
        """Close the environment."""
        pass


# Register the environment
gym.register(
    id='UltimateTicTacToe-v0',
    entry_point='ultimate_tic_tac_toe.env:UltimateTicTacToeEnv',
    max_episode_steps=81,  # Maximum possible moves
) 