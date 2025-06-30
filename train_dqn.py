#!/usr/bin/env python3
"""
DQN Training Script for Ultimate Tic-Tac-Toe
Uses multi-plane tensor state representation similar to AlphaGo.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
import json
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv
from ultimate_tic_tac_toe.agent_impl import RandomAgent
from ultimate_tic_tac_toe.dqn_trainer import DQNAgent, DQNTrainer, DQNTrainingAlgorithm
from ultimate_tic_tac_toe.encoder_impl import EncoderFactory


class TrainingConfig:
    """Configuration class for DQN training."""
    
    def __init__(self, **kwargs):
        # Environment settings
        self.sovereignty_upon_draw = kwargs.get('sovereignty_upon_draw', 'none')
        
        # Encoder settings
        self.encoder_type = kwargs.get('encoder_type', 'multiplane')
        
        # DQN hyperparameters
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.memory_size = kwargs.get('memory_size', 10000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.target_update_freq = kwargs.get('target_update_freq', 1000)
        
        # Training settings
        self.num_episodes = kwargs.get('num_episodes', 10000)
        self.evaluation_interval = kwargs.get('evaluation_interval', 100)
        self.save_interval = kwargs.get('save_interval', 1000)
        self.log_interval = kwargs.get('log_interval', 10)
        
        # Model settings
        self.model_save_path = kwargs.get('model_save_path', 'models/dqn')
        self.device = kwargs.get('device', 'auto')  # 'auto', 'cpu', or 'cuda'
        
        # Evaluation settings
        self.eval_games = kwargs.get('eval_games', 100)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            'sovereignty_upon_draw': self.sovereignty_upon_draw,
            'encoder_type': self.encoder_type,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'num_episodes': self.num_episodes,
            'evaluation_interval': self.evaluation_interval,
            'save_interval': self.save_interval,
            'log_interval': self.log_interval,
            'model_save_path': self.model_save_path,
            'device': self.device,
            'eval_games': self.eval_games
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class TrainingCallback:
    """Base class for training callbacks."""
    
    def __call__(self, trainer: DQNTrainer, episode: int, metrics: Dict[str, Any]):
        """Called after each training episode."""
        pass


class ProgressPlotter(TrainingCallback):
    """Callback for plotting training progress."""
    
    def __init__(self, plot_interval: int = 100):
        self.plot_interval = plot_interval
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.epsilons = []
        self.win_rates = []
        
    def __call__(self, trainer: DQNTrainer, episode: int, metrics: Dict[str, Any]):
        """Update plots after each episode."""
        self.episodes.append(episode)
        self.rewards.append(metrics['total_reward'])
        self.losses.append(metrics['avg_loss'])
        self.epsilons.append(metrics['epsilon'])
        
        # Add win rate if available from evaluation
        if trainer.evaluation_history:
            latest_eval = trainer.evaluation_history[-1]
            self.win_rates.append(latest_eval['win_rate'])
        else:
            self.win_rates.append(0.0)
        
        # Plot every plot_interval episodes
        if episode % self.plot_interval == 0:
            self._plot_progress()
    
    def _plot_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot rewards
        axes[0, 0].plot(self.episodes, self.rewards, alpha=0.6)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Plot losses
        axes[0, 1].plot(self.episodes, self.losses, alpha=0.6, color='red')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        
        # Plot epsilon
        axes[1, 0].plot(self.episodes, self.epsilons, alpha=0.6, color='green')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Plot win rates
        axes[1, 1].plot(self.episodes, self.win_rates, alpha=0.6, color='purple')
        axes[1, 1].set_title('Win Rate vs Random')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()


class ModelSaver(TrainingCallback):
    """Callback for saving model checkpoints."""
    
    def __init__(self, save_path: str, save_interval: int = 1000):
        self.save_path = save_path
        self.save_interval = save_interval
        os.makedirs(save_path, exist_ok=True)
    
    def __call__(self, trainer: DQNTrainer, episode: int, metrics: Dict[str, Any]):
        """Save model checkpoint."""
        if episode % self.save_interval == 0:
            checkpoint_path = os.path.join(self.save_path, f'checkpoint_episode_{episode}.pth')
            trainer.agent.save_model(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


def setup_device(device_preference: str) -> str:
    """Setup and return the appropriate device."""
    if device_preference == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return device_preference


def create_training_components(config: TrainingConfig):
    """Create all components needed for training."""
    
    # Setup device
    device = setup_device(config.device)
    print(f"Using device: {device}")
    
    # Create environment
    env = UltimateTicTacToeEnv(
        sovereignty_upon_draw=config.sovereignty_upon_draw,
        render_mode=None
    )
    
    # Create encoder
    encoder = EncoderFactory.create_encoder(config.encoder_type)
    print(f"Using encoder: {type(encoder).__name__}")
    print(f"Input shape: {encoder.get_input_shape()}")
    
    # Create agents
    dqn_agent = DQNAgent(
        env=env,
        encoder=encoder,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        device=device
    )
    
    opponent = RandomAgent(env)
    
    # Create training algorithm
    training_algorithm = DQNTrainingAlgorithm(
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        gamma=config.gamma,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min
    )
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        agent=dqn_agent,
        opponent=opponent,
        encoder=encoder,
        training_algorithm=training_algorithm,
        evaluation_interval=config.evaluation_interval,
        save_interval=config.save_interval,
        model_save_path=config.model_save_path,
        log_interval=config.log_interval
    )
    
    return trainer, dqn_agent, encoder


def train_dqn(config: TrainingConfig, callbacks: Optional[List[TrainingCallback]] = None):
    """Main training function."""
    
    print("=" * 60)
    print("DQN Training for Ultimate Tic-Tac-Toe")
    print("=" * 60)
    
    # Save configuration
    config_path = os.path.join(config.model_save_path, 'config.json')
    config.save(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Create training components
    trainer, dqn_agent, encoder = create_training_components(config)
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks
    callbacks.append(ProgressPlotter(plot_interval=config.evaluation_interval))
    callbacks.append(ModelSaver(config.model_save_path, config.save_interval))
    
    # Start training
    start_time = time.time()
    
    try:
        results = trainer.train(config.num_episodes, callbacks=callbacks)
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = os.path.join(config.model_save_path, 'final_model.pth')
        dqn_agent.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Print final results
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Episodes trained: {results['episodes_trained']}")
        print(f"Final evaluation results:")
        final_eval = results['final_evaluation']
        print(f"  Win Rate: {final_eval['win_rate']:.3f}")
        print(f"  Loss Rate: {final_eval['loss_rate']:.3f}")
        print(f"  Draw Rate: {final_eval['draw_rate']:.3f}")
        print(f"  Average Reward: {final_eval['avg_reward']:.3f}")
        
        # Save training results
        results_path = os.path.join(config.model_save_path, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                'episodes_trained': results['episodes_trained'],
                'training_time': results['training_time'],
                'final_evaluation': results['final_evaluation'],
                'training_history': [
                    {k: float(v) if isinstance(v, np.number) else v 
                     for k, v in episode.items()}
                    for episode in results['training_history']
                ],
                'evaluation_history': [
                    {k: float(v) if isinstance(v, np.number) else v 
                     for k, v in eval_result.items()}
                    for eval_result in results['evaluation_history']
                ]
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training results saved to: {results_path}")
        
        return results
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save intermediate model
        interrupt_model_path = os.path.join(config.model_save_path, 'interrupted_model.pth')
        dqn_agent.save_model(interrupt_model_path)
        print(f"Intermediate model saved to: {interrupt_model_path}")
        return None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train DQN agent for Ultimate Tic-Tac-Toe')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--memory-size', type=int, default=10000, help='Replay memory size')
    
    # Model parameters
    parser.add_argument('--encoder', type=str, default='multiplane', 
                       choices=['multiplane', 'simple'], help='State encoder type')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    # Training settings
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=1000, help='Model save interval')
    parser.add_argument('--log-interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--model-path', type=str, default='models/dqn', help='Model save path')
    
    # Game settings
    parser.add_argument('--sovereignty', type=str, default='none', 
                       choices=['none', 'both'], help='Sovereignty upon draw rule')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        encoder_type=args.encoder,
        device=args.device,
        evaluation_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        model_save_path=args.model_path,
        sovereignty_upon_draw=args.sovereignty
    )
    
    # Start training
    train_dqn(config)


if __name__ == "__main__":
    main() 