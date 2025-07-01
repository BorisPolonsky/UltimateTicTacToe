#!/usr/bin/env python3
"""
Example script showing how to use the Ultimate Tic-Tac-Toe environment for training.
This demonstrates a simple random agent and how to interact with the environment.
"""

import numpy as np
import gymnasium as gym
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv
from ultimate_tic_tac_toe.agent_impl import RandomAgent


def play_game(env, agent1, agent2, render=False):
    """Play a complete game between two agents."""
    observation, info = env.reset()
    total_reward = 0
    step_count = 0
    terminated = False
    
    while not terminated:
        # Determine current agent based on info from environment
        current_agent = agent1 if info['next_player'] == 1 else agent2
        
        # Get action from current agent
        action = current_agent.get_action(observation, info)
        
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
                    f"Valid moves count: {np.sum(env.get_valid_actions())}\n"
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
        
        if terminated:
            if render:
                if info['winner'] == 0:
                    print("Game ended in a draw!")
                else:
                    print(f"Player {info['winner']} wins!")
    
    return total_reward, step_count, info['winner']


def train_agents(episodes=1000):
    """Train two agents against each other."""
    print("Training agents...")
    
    # Training loop
    for episode in range(episodes):
        env = UltimateTicTacToeEnv()
        agent1 = RandomAgent(env)
        agent2 = RandomAgent(env)
        observation, info = env.reset()
        
        while not info['game_over']:
            # Determine current agent based on next_player
            current_agent = agent1 if info['next_player'] == 1 else agent2
            
            # Get action from current agent
            action = current_agent.get_action(observation, info)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    print("Training completed!")


def demonstrate_environment():
    """Demonstrate the environment with a single game."""
    print("Demonstrating Environment...")
    
    # Create environment
    env = UltimateTicTacToeEnv(render_mode="human")
    
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
    """Test that the environment is properly registered with gymnasium."""
    print("Testing Environment Registration...")
    
    try:
        # Try to create environment using gymnasium.make
        env = gym.make('UltimateTicTacToe-v0')
        print("✅ Environment successfully registered with gymnasium!")
        
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


def test_trained_agents():
    """Test the trained agents."""
    print("Testing trained agents...")
    
    # Create environment
    env = UltimateTicTacToeEnv(render_mode="human")
    
    # Create agents (in a real scenario, these would be trained agents)
    agent1 = RandomAgent(env)
    agent2 = RandomAgent(env)
    
    # Play a game
    observation, info = env.reset()
    env.render()
    
    while not info['game_over']:
        # Determine current agent
        current_agent = agent1 if info['next_player'] == 1 else agent2
        
        # Get action
        action = current_agent.get_action(observation, info)
        
        # Take step
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            break
    
    # Print result
    if info['winner'] == 0:
        print("Game ended in a draw!")
    elif info['winner'] == 1:
        print("Player 1 wins!")
    else:
        print("Player 2 wins!")
    
    env.close()


if __name__ == "__main__":
    print("Ultimate Tic-Tac-Toe - Training Example")
    print("=" * 50)
    
    # Test environment registration
    test_environment_registration()
    
    print("\n" + "=" * 50)
    
    # Demonstrate environment
    demonstrate_environment()
    
    print("\n" + "=" * 50)
    
    # Train agents
    train_agents(episodes=100)
    
    print("\n" + "=" * 50)
    
    # Test trained agents
    test_trained_agents()
    
    print("\n" + "=" * 50)
    print("Example completed! 🎉") 