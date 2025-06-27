#!/usr/bin/env python3
"""
Example script showing how to use the Ultimate Tic-Tac-Toe environment for training.
This demonstrates a simple random agent and how to interact with the environment.
"""

import numpy as np
import gym
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv


class RandomAgent:
    """Simple random agent for demonstration."""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self, observation, valid_actions):
        """Choose a random valid action."""
        valid_action_indices = np.where(valid_actions == 1)[0]
        if len(valid_action_indices) == 0:
            return None
        return np.random.choice(valid_action_indices)


def play_game(env, agent1, agent2, render=False):
    """Play a complete game between two agents."""
    observation, info = env.reset()
    total_reward = 0
    step_count = 0
    terminated = False
    
    while not terminated:
        # Determine current agent based on info from environment
        current_agent = agent1 if info['next_player'] == env.initiator else agent2
        
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Get action from current agent
        action = current_agent.get_action(observation, valid_actions)
        
        if action is None:
            # Check if the game should be terminated based on info
            if not info['game_over']:
                # This is a bug - no valid actions but game isn't over
                error_msg = (
                    f"Environment bug detected! No valid actions available but game is not over.\n"
                    f"Step: {step_count}\n"
                    f"Next player: {info['next_player']}\n"
                    f"Next block: {info['next_block']}\n"
                    f"Game over: {info['game_over']}\n"
                    f"Winner: {info['winner']}\n"
                    f"Valid moves count: {np.sum(valid_actions)}\n"
                    f"Board state:\n{env.board}"
                )
                raise RuntimeError(error_msg)
            else:
                # Game is over, no valid actions expected
                print("Game ended - no valid actions available.")
                break
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if render:
            print(f"Step {step_count}: Player {info['next_player']} (3-{info['next_player']})")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            env.render()
            print()
        
    if render:
        if info['winner'] == 0:
            print("Game ended in a draw!")
        else:
            print(f"Player {info['winner']} wins!")
    
    return total_reward, step_count, info['winner']


def train_random_agents(num_episodes=1000):
    """Train random agents and collect statistics."""
    print("Training Random Agents...")
    
    # Create environment
    env = UltimateTicTacToeEnv(initiator=1)
    
    # Create agents
    agent1 = RandomAgent(env)
    agent2 = RandomAgent(env)
    
    # Statistics
    wins_player1 = 0
    wins_player2 = 0
    draws = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}")
        
        # Play game
        reward, steps, winner = play_game(env, agent1, agent2, render=False)
        
        # Update statistics
        total_steps += steps
        if winner == 1:
            wins_player1 += 1
        elif winner == 2:
            wins_player2 += 1
        else:
            draws += 1
    
    # Print results
    print(f"\nTraining Results ({num_episodes} episodes):")
    print(f"Player 1 wins: {wins_player1} ({wins_player1/num_episodes*100:.1f}%)")
    print(f"Player 2 wins: {wins_player2} ({wins_player2/num_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.1f}%)")
    print(f"Average steps per game: {total_steps/num_episodes:.1f}")
    
    env.close()


def demonstrate_environment():
    """Demonstrate the environment with a single game."""
    print("Demonstrating Environment...")
    
    # Create environment
    env = UltimateTicTacToeEnv(initiator=1, render_mode="human")
    
    # Create agents
    agent1 = RandomAgent(env)
    agent2 = RandomAgent(env)
    
    # Play a single game with rendering
    print("Playing a demonstration game:")
    reward, steps, winner = play_game(env, agent1, agent2, render=True)
    
    print(f"\nGame Summary:")
    print(f"Total steps: {steps}")
    print(f"Total reward: {reward}")
    print(f"Winner: {winner}")
    
    env.close()


def test_environment_registration():
    """Test that the environment is properly registered with gym."""
    print("Testing Environment Registration...")
    
    try:
        # Try to create environment using gym.make
        env = gym.make('UltimateTicTacToe-v0')
        print("✅ Environment successfully registered with gym!")
        
        # Test basic functionality
        observation, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {observation.shape}")
        
        # Test action space
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful - Reward: {reward}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Environment registration failed: {e}")


if __name__ == "__main__":
    print("Ultimate Tic-Tac-Toe - Training Example")
    print("=" * 50)
    
    # Test environment registration
    test_environment_registration()
    
    print("\n" + "=" * 50)
    
    # Demonstrate environment
    demonstrate_environment()
    
    print("\n" + "=" * 50)
    
    # Train random agents
    train_random_agents(num_episodes=100)
    
    print("\n" + "=" * 50)
    print("Example completed! 🎉") 