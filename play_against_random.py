#!/usr/bin/env python3
"""
Interactive script to play Ultimate Tic-Tac-Toe against a random AI agent.
"""

import numpy as np
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv


class RandomAgent:
    """Simple random agent."""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self, valid_actions):
        """Choose a random valid action."""
        valid_action_indices = np.where(valid_actions == 1)[0]
        if len(valid_action_indices) == 0:
            return None
        return np.random.choice(valid_action_indices)


def print_board_with_coordinates():
    """Print a reference board showing coordinates."""
    print("\nBoard Coordinates Reference:")
    print("Each move is specified as: block_row,block_col,slot_row,slot_col")
    print("Blocks are numbered 0-2 from top to bottom, left to right")
    print("Slots within each block are numbered 0-2 from top to bottom, left to right")
    print()
    
    for block_row in range(3):
        for slot_row in range(3):
            row_parts = []
            for block_col in range(3):
                block_parts = []
                for slot_col in range(3):
                    coord = f"{block_row},{block_col},{slot_row},{slot_col}"
                    block_parts.append(coord)
                row_parts.append(" | ".join(block_parts))
            print(" " + "||".join(row_parts) + " ")
        
        if block_row < 2:
            print("-" * 100)


def get_human_move(env):
    """Get a move from the human player."""
    valid_actions = env.get_valid_actions()
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
            action = env._encode_action(block_row, block_col, slot_row, slot_col)
            
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


def play_game():
    """Play a complete game against the random agent."""
    print("Ultimate Tic-Tac-Toe - Play Against Random AI")
    print("=" * 50)
    print("You are Player 1 (X), AI is Player 2 (O)")
    
    # Create environment
    env = UltimateTicTacToeEnv(initiator=1, render_mode="human")
    
    # Create random agent
    ai_agent = RandomAgent(env)
    
    # Reset environment
    observation, info = env.reset()
    
    print_board_with_coordinates()
    
    step_count = 0
    
    while True:
        step_count += 1
        current_player = env.board.next_player
        
        print(f"\n{'='*50}")
        print(f"Step {step_count}: Player {current_player} ({'You' if current_player == 1 else 'AI'})")
        print(f"Next block: {env.board.next_block}")
        
        # Render current state
        env.render()
        
        if current_player == 1:  # Human player
            action = get_human_move(env)
            if action is None:
                print("Game ended by user.")
                break
        else:  # AI player
            print("AI is thinking...")
            action = ai_agent.get_action(env.get_valid_actions())
            if action is None:
                print("AI has no valid moves!")
                break
            
            # Decode action for display
            block_row, block_col, slot_row, slot_col = env._decode_action(action)
            print(f"AI plays: ({block_row},{block_col},{slot_row},{slot_col})")
        
        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"\n{'='*50}")
            print("Game Over!")
            env.render()
            
            if info['winner'] == 0:
                print("It's a draw!")
            elif info['winner'] == 1:
                print("Congratulations! You win! 🎉")
            else:
                print("AI wins! Better luck next time!")
            
            break
    
    env.close()
    print(f"\nGame completed in {step_count} steps.")


def main():
    """Main function."""
    while True:
        play_game()
        
        play_again = input("\nPlay again? (y/n): ").strip().lower()
        if play_again != 'y':
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main() 