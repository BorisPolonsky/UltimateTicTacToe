# DQN Training for Ultimate Tic-Tac-Toe

This repository implements a Deep Q-Network (DQN) training system for Ultimate Tic-Tac-Toe using a multi-plane tensor state representation similar to AlphaGo.

## Features

- **Multi-Plane Tensor Encoding**: State representation using 3x3x23 tensors with 23 different planes capturing various aspects of the game state
- **Clean Architecture**: Clear separation between agents (action selection) and training algorithms (learning logic)
- **Modular Design**: Easy to swap encoders, models, and training algorithms
- **Flexible Configuration**: Comprehensive configuration system for hyperparameters
- **Progress Tracking**: Real-time plotting and logging of training progress
- **Model Checkpointing**: Automatic model saving and loading
- **Evaluation System**: Regular evaluation against random opponents

## Architecture Overview

### Clean Separation of Concerns

1. **Agents**: Handle action selection during gameplay only
   - `Agent` (abstract): Base class for all agents
   - `DQNAgent`: Neural network-based action selection with epsilon-greedy exploration
   - `RandomAgent`: Random action selection for opponents
   - `HumanAgent`: Human input for interactive play

2. **Training Algorithms**: Handle all training logic
   - `TrainingAlgorithm` (abstract): Base class for training algorithms
   - `DQNTrainingAlgorithm`: Experience replay, network updates, epsilon decay
   - Extensible for other algorithms (A3C, PPO, etc.)

3. **Encoders**: Convert game state to neural network input
   - `StateEncoder` (abstract): Base class for state encoders
   - `MultiPlaneEncoder`: 3x3x23 tensor representation (recommended)
   - `SimpleEncoder`: Flattened 81-dimensional vector (baseline)

4. **Trainer**: Orchestrates the overall training process
   - Manages episodes, evaluation, logging, and callbacks

### File Structure

```
ultimate_tic_tac_toe/
├── abstract_agent.py      # Abstract Agent class
├── agent_impl.py          # Concrete agent implementations (RandomAgent, HumanAgent)
├── dqn_agent.py           # DQN agent implementation
├── abstract_encoders.py   # Abstract StateEncoder class
├── encoder_impl.py        # Concrete encoder implementations (MultiPlaneEncoder, SimpleEncoder)
├── abstract_trainer.py    # Abstract TrainingAlgorithm class
├── trainer_impl.py        # Concrete trainer implementations (DQNTrainingAlgorithm, Trainer)
├── env.py                 # OpenAI Gym environment
└── board.py               # Game board logic
```

## Multi-Plane Tensor Representation

The game state is encoded as a 3x3x23 tensor with the following planes:

### Local Board States (Planes 0-17)
- **Planes 0-8**: Player X's move density in each small board (0.0 to 1.0)
- **Planes 9-17**: Player O's move density in each small board (0.0 to 1.0)

### Global Board State (Planes 18-20)
- **Plane 18**: Small boards captured by Player X (binary: 0/1)
- **Plane 19**: Small boards captured by Player O (binary: 0/1)
- **Plane 20**: Draw/stalemate small boards (binary: 0/1)

### Game State (Planes 21-22)
- **Plane 21**: Valid move density per block (0.0 to 1.0)
- **Plane 22**: Current player indicator (0.0 for X, 1.0 for O)

## Installation

1. Install the required dependencies:
```bash
pip install torch numpy matplotlib gym
```

2. Make sure the `ultimate_tic_tac_toe` package is in your Python path.

## Quick Start

### Basic Training

Run a quick training session with default settings:

```bash
python train_dqn.py --episodes 1000 --encoder multiplane --device auto
```

### Command Line Training

Use the command-line interface for custom training:

```bash
python train_dqn.py --episodes 5000 --encoder multiplane --device auto
```

### Available Command Line Options

- `--episodes`: Number of training episodes (default: 10000)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-min`: Minimum exploration rate (default: 0.01)
- `--epsilon-decay`: Epsilon decay rate (default: 0.995)
- `--batch-size`: Batch size for training (default: 32)
- `--memory-size`: Replay memory size (default: 10000)
- `--encoder`: State encoder type ('multiplane' or 'simple', default: 'multiplane')
- `--device`: Device to use ('auto', 'cpu', or 'cuda', default: 'auto')
- `--eval-interval`: Evaluation interval (default: 100)
- `--save-interval`: Model save interval (default: 1000)
- `--log-interval`: Logging interval (default: 10)
- `--model-path`: Model save path (default: 'models/dqn')
- `--sovereignty`: Sovereignty upon draw rule ('none' or 'both', default: 'none')

## Programmatic Usage

### Basic Training

```python
from train_dqn import TrainingConfig, train_dqn

# Create configuration
config = TrainingConfig(
    num_episodes=1000,
    encoder_type='multiplane',
    learning_rate=1e-4,
    batch_size=32
)

# Start training
results = train_dqn(config)
```

### Custom Configuration

```python
config = TrainingConfig(
    # Training parameters
    num_episodes=5000,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    memory_size=20000,
    
    # Encoder and model
    encoder_type='multiplane',
    device='cuda',  # Use GPU if available
    
    # Training intervals
    evaluation_interval=100,
    save_interval=500,
    log_interval=20,
    
    # Model path
    model_save_path='models/my_dqn_model',
    
    # Game settings
    sovereignty_upon_draw='none'
)
```

## Components

### Agents

**DQNAgent**: Neural network-based agent
- Epsilon-greedy action selection
- Neural network for Q-value estimation
- Training mode toggle for evaluation

**RandomAgent**: Simple random opponent
- Random action selection from valid moves
- Used for training and evaluation

### Training Algorithms

**DQNTrainingAlgorithm**: Deep Q-Network implementation
- Experience replay buffer
- Batch training with target network
- Epsilon decay management
- Network updates and loss calculation

### State Encoders

- **MultiPlaneEncoder**: 3x3x23 tensor representation (recommended)
- **SimpleEncoder**: Flattened 81-dimensional vector (baseline)

### Neural Network Architecture

For multi-plane input (3x3x23):
- 3 convolutional layers with batch normalization
- 2 fully connected layers
- Output: 81 Q-values (one per action)

For simple input (81):
- 3 fully connected layers
- Output: 81 Q-values (one per action)

## Training Process

1. **Episode Loop**: Each episode consists of a complete game
2. **Action Selection**: Agents select actions using their policies
3. **Experience Collection**: Training algorithm stores experiences
4. **Training**: Sample batches from replay memory and update Q-network
5. **Target Network**: Periodically update target network for stability
6. **Evaluation**: Regular evaluation against random opponent

## Output Files

Training generates several output files:

- `config.json`: Training configuration
- `training_results.json`: Complete training history and results
- `final_model.pth`: Final trained model
- `checkpoint_episode_X.pth`: Model checkpoints
- `training_progress.png`: Training progress plots

## Monitoring Training

### Progress Plots

The system automatically generates plots showing:
- Training rewards over time
- Loss values during training
- Exploration rate (epsilon) decay
- Win rate against random opponent

### Logging

Training progress is logged to console with metrics including:
- Episode number
- Total reward
- Number of steps
- Winner
- Average loss
- Current epsilon value

## Evaluation

The system evaluates the trained agent by:
- Playing 100 games against a random opponent
- Computing win rate, loss rate, and draw rate
- Calculating average reward and standard deviation

## Customization

### Adding New Agents

```python
from ultimate_tic_tac_toe.abstract_agent import Agent

class CustomAgent(Agent):
    def get_action(self, observation, info):
        # Your action selection logic here
        return action
    
    def set_training_mode(self, training):
        # Handle training mode if needed
        pass
```

### Adding New Training Algorithms

```python
from ultimate_tic_tac_toe.abstract_trainer import TrainingAlgorithm

class CustomTrainingAlgorithm(TrainingAlgorithm):
    def __init__(self, **kwargs):
        # Initialize your algorithm
        pass
    
    def train_episode(self, env, agent, opponent, encoder):
        # Your training logic here
        return metrics
```

### Adding New Encoders

```python
from ultimate_tic_tac_toe.abstract_encoders import StateEncoder

class CustomEncoder(StateEncoder):
    def encode(self, observation, info):
        # Your encoding logic here
        return encoded_state
    
    def get_input_shape(self):
        return (your_shape,)
```

### Custom Callbacks

```python
from train_dqn import TrainingCallback

class CustomCallback(TrainingCallback):
    def __call__(self, trainer, episode, metrics):
        # Your callback logic here
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: Use GPU if available, reduce evaluation frequency
3. **Poor Performance**: Increase training episodes, adjust learning rate

### Performance Tips

- Use GPU for faster training
- Increase replay memory size for better stability
- Adjust epsilon decay rate based on training progress
- Use larger batch sizes if memory allows

## Example Results

A typical training run might achieve:
- Win rate: 60-80% against random opponent
- Training time: 1-4 hours for 10,000 episodes
- Final epsilon: ~0.01 (mostly exploitation)

## License

This project is part of the Ultimate Tic-Tac-Toe implementation.