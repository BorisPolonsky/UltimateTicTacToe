#!/usr/bin/env python3
"""
Interactive script to play Ultimate Tic-Tac-Toe against a random AI agent.
"""

import numpy as np
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv
from ultimate_tic_tac_toe.agent import RandomAgent, HumanAgent


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


def play_game():
    """Play a complete game against the random agent."""
    print("Ultimate Tic-Tac-Toe - Play Against Random AI")
    print("=" * 50)
    print("You are Player 1 (X), AI is Player 2 (O)")
    
    # Create environment
    env = UltimateTicTacToeEnv(initiator=1, render_mode="human")
    
    # Create agents
    human_agent = HumanAgent(env)
    ai_agent = RandomAgent(env)
    
    # Reset environment
    observation, info = env.reset()
    
    print_board_with_coordinates()
    
    step_count = 0
    terminated = False
    
    while not terminated:
        step_count += 1
        current_player = info['next_player']
        
        print(f"\n{'='*50}")
        print(f"Step {step_count}: Player {current_player} ({'You' if current_player == 1 else 'AI'})")
        print(f"Next block: {info['next_block']}")
        
        # Render current state
        env.render()
        
        if current_player == 1:  # Human player
            action = human_agent.get_action(observation, info)
            if action is None:
                print("Game ended by user.")
                break
        else:  # AI player
            print("AI is thinking...")
            action = ai_agent.get_action(observation, info)
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