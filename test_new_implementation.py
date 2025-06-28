#!/usr/bin/env python3
"""
Test script for the new Ultimate Tic-Tac-Toe implementation.
Demonstrates the Board class and Gym environment.
"""

import numpy as np
from ultimate_tic_tac_toe.board import Board
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv


def test_board_class():
    """Test the new Board class."""
    print("=== Testing Board Class ===")
    
    # Create a new board
    board = Board(sovereignty_upon_draw="none")
    print("Initial board:")
    print(board)
    print(f"Next player: {board.next_player}")
    print(f"Valid moves: {len(board.get_valid_moves())}")
    
    # Make some moves
    print("\nMaking moves...")
    
    # Move 1: Player 1 plays in top-left block, center slot
    game_over = board.make_move(0, 0, 1, 1)
    print(f"Move 1 - Player 1 plays (0,0,1,1):")
    print(board)
    print(f"Game over: {game_over}")
    print(f"Next player: {board.next_player}")
    print(f"Next block: {board.next_block}")
    
    # Move 2: Player 2 must play in center block (1,1)
    game_over = board.make_move(1, 1, 0, 0)
    print(f"\nMove 2 - Player 2 plays (1,1,0,0):")
    print(board)
    print(f"Game over: {game_over}")
    print(f"Next player: {board.next_player}")
    print(f"Next block: {board.next_block}")
    
    # Get valid moves
    valid_moves = board.get_valid_moves()
    print(f"\nValid moves: {valid_moves[:5]}... (showing first 5)")
    
    # Test state representation
    state = board.get_state()
    print(f"\nBoard shape: {state['board'].shape}")
    print(f"Block status shape: {state['block_status'].shape}")
    print(f"Game over: {state['game_over']}")
    print(f"Winner: {state['winner']}")


def test_gym_environment():
    """Test the Gym environment."""
    print("\n=== Testing Gym Environment ===")
    
    # Create environment
    env = UltimateTicTacToeEnv(render_mode="human")
    
    # Reset environment
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Get valid actions
    valid_actions = env.get_valid_actions()
    print(f"Number of valid actions: {np.sum(valid_actions)}")
    
    # Make a few random moves
    print("\nMaking random moves...")
    for step in range(5):
        # Get valid actions
        valid_actions = env.get_valid_actions()
        valid_action_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_action_indices) == 0:
            print("No valid actions available!")
            break
        
        # Choose random valid action
        action = np.random.choice(valid_action_indices)
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Winner: {info['winner']}")
        
        if terminated:
            print("Game ended!")
            break
    
    # Render final state
    print("\nFinal board:")
    env.render()
    
    env.close()


def test_action_encoding():
    """Test action encoding and decoding."""
    print("=== Testing Action Encoding/Decoding ===")
    
    env = UltimateTicTacToeEnv()
    
    # Test some moves with the new encoding
    # Action index = row * 9 + col
    test_moves = [
        (0, 0, 0, 0),  # block (0,0), slot (0,0) -> board (0,0) -> action 0
        (0, 0, 1, 1),  # block (0,0), slot (1,1) -> board (1,1) -> action 10
        (1, 1, 2, 2),  # block (1,1), slot (2,2) -> board (5,5) -> action 50
        (2, 2, 0, 0),  # block (2,2), slot (0,0) -> board (6,6) -> action 60
    ]
    
    for move in test_moves:
        block_row, block_col, slot_row, slot_col = move
        action = env._encode_action(block_row, block_col, slot_row, slot_col)
        decoded = env._decode_action(action)
        
        print(f"Move {move} -> Action {action} -> Decoded {decoded}")
        assert decoded == move, f"Encoding/decoding failed for {move}"
    
    print("Action encoding/decoding test passed!")


def test_basic_game():
    """Test basic game functionality."""
    board = Board(sovereignty_upon_draw="none")
    
    # Test initial state
    assert board.next_player == 1
    assert not board.game_over
    assert board.winner is None
    assert board.next_block is None
    
    # Test first move
    assert board.is_valid_move(0, 0, 0, 0)
    game_over = board.make_move(0, 0, 0, 0)
    assert not game_over
    assert board.next_player == 2
    assert board.next_block == (0, 0)
    
    print("Basic game test passed!")


def test_board_copy():
    """Test board copying functionality."""
    board1 = Board()
    
    # Make some moves
    board1.make_move(0, 0, 0, 0)
    board1.make_move(0, 0, 0, 1)
    
    # Copy the board
    board2 = board1.copy()
    
    # Verify they're identical
    assert np.array_equal(board1.board, board2.board)
    assert np.array_equal(board1.block_status, board2.block_status)
    assert board1.next_player == board2.next_player
    assert board1.next_block == board2.next_block
    assert board1.game_over == board2.game_over
    assert board1.winner == board2.winner
    
    # Verify they're independent - make a valid move on the copy
    # The next_block should be (0,0) after the previous moves
    if board2.next_block == (0, 0):
        board2.make_move(0, 0, 0, 2)  # This should be valid
        assert not np.array_equal(board1.board, board2.board)
    else:
        # If next_block is None, we can play anywhere
        valid_moves = board2.get_valid_moves()
        if valid_moves:
            move = valid_moves[0]
            board2.make_move(*move)
            assert not np.array_equal(board1.board, board2.board)
    
    print("Board copy test passed!")


if __name__ == "__main__":
    print("Ultimate Tic-Tac-Toe - New Implementation Test")
    print("=" * 50)
    
    try:
        test_board_class()
        test_gym_environment()
        test_action_encoding()
        test_basic_game()
        test_board_copy()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
