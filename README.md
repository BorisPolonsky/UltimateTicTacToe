# UltimateTicTacToe

## What is this?
This is a Python3 implementation of Ultimate Tic-Tac-Toe with both a traditional game interface and a modern reinforcement learning environment. Check this [link](https://mathwithbaddrawings.com/2013/06/16/ultimate-tic-tac-toe/) for details about the game rules.

## New Refactored Implementation

This project has been refactored to provide a modern, efficient implementation suitable for solving UTTT with reinforcement learning:

### Key Features

1. **Board Class (`ultimate_tic_tac_toe/board.py`)**:
   - Represents the board as a 9x9 numpy array
   - 0 = unoccupied, 1 = player 1, 2 = player 2, 3 = draw
   - Efficient move validation and game state tracking
   - Support for both rule variants (normal and bizarre)

2. **OpenAI Gym Environment (`ultimate_tic_tac_toe/env.py`)**:
   - Standard RL interface with discrete action space (81 actions)
   - 9x9 observation space representing board state
   - Proper reward structure for training
   - Action masking for valid moves
   - Render support for visualization

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```python
from ultimate_tic_tac_toe.board import Board
from ultimate_tic_tac_toe.env import UltimateTicTacToeEnv

# Use the Board class directly
board = Board()
game_over = board.make_move(0, 0, 1, 1)  # block_row, block_col, slot_row, slot_col
print(board)

# Use as a Gym environment
env = UltimateTicTacToeEnv()
observation, info = env.reset()
action = env.action_space.sample()  # Random action
observation, reward, terminated, truncated, info = env.step(action)
```

### Testing

Run the test script to verify the implementation:

```bash
python test_new_implementation.py
```

### Training Example

See `example_training.py` for a complete example of how to use the environment for training agents.

## Original Implementation

The original implementation is still available in the `ultimate_tic_tac_toe/` directory:

- `game_board.py`: Original string-based board implementation
- `game_manager.py`: Game management and terminal interface
- `mcts.py`: Monte Carlo Tree Search implementation

## Algorithm
* [MCTS & UCT](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (original implementation)
* Reinforcement Learning ready (new implementation)

## "Terminologies"
A total of 2 rule sets are supported in this program, according to this statement in the [original post](https://mathwithbaddrawings.com/2013/06/16/ultimate-tic-tac-toe/). 
> What if one of the small boards results in a tie? I recommend that the board counts for neither X nor O. But, if you feel like a crazy variant, you could agree before the game to count a tied board for both X and O. 

In this program, the rule set recommended in the post is refered as **normal**, while the other is refered as **bizarre**. 

## TODO
To enhance the unsatisfactory performance (e.g. competance, computational cost) of the program for now, the following tasks are still in progress

* Train proper models with better performance for both rule sets. 
* Optimize the algorithm for MCTS.
* Clean up the code.
* Implement state-evaluation function/network. 
* Utilize dynamic computational cost. 
* Implement advanced RL algorithms (PPO, A3C, etc.)
* Add self-play training capabilities
