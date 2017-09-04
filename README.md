# UltimateTicTacToe
## What is this?
This is a terminal-based AI test program in Python3 for an extended tic-tac-toe game. Check this [link](https://mathwithbaddrawings.com/2013/06/16/ultimate-tic-tac-toe/) for details and other implementations.

## Algorithm
* [MCTS & UCT](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

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
