import pickle
import time
import os
import math
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe
import random
from enum import Enum, unique
import copy


class MCT:
    class NodeMCT:
        @unique
        class Result(Enum):
            the_initiator_wins = 0
            the_initiator_loses = 1
            draw = 2

        def __init__(self, op=None):
            """
            Initialization of a node.
            self._op: the operation from last state to reach this node.
            None for root, (row_block, row_column, row_slot, column_slot) for the rest.
            self._val: a tuple of (#game in which the initiator wins, #game in which the initiator loses, #total games)
            #draw=self._val[2]-self._val[0]-self._val[1]
            self._next: a list of children of the node.
            """
            if type(op) == tuple and len(op) == 4:
                for val in op:
                    if val not in range(1, 4):
                        raise ValueError("Each entry of the state must be an integer from 1-3")
                self._op = op
            elif op is None:
                self._op = op
            else:
                raise TypeError('Parameter "op" must be either None or a tuple of (row_block, row_column, row_slot, column_slot)')
            self._val = (0, 0, 0)
            self._next = []

        # Need to be checked
        def add_child(self, action):
            node = MCT.NodeMCT(action)
            self._next.append(node)
            return node

        def update(self, result):
            if result == MCT.NodeMCT.Result.the_initiator_wins:
                self._val = (self._val[0] + 1, self._val[1], self._val[2] + 1)
            elif result == MCT.NodeMCT.Result.the_initiator_loses:
                self._val = (self._val[0], self._val[1] + 1, self._val[2] + 1)
            elif result == MCT.NodeMCT.Result.draw:
                self._val = (self._val[0], self._val[1], self._val[2] + 1)  # ?
            else:
                raise ValueError("Invalid result. ")

        def get_best_child(self, is_initiator, c=math.sqrt(2)):
            if c < 0:
                raise ValueError('Parameter c must be greater or equal to 0. ')
            children = self.children
            score_node_pairs = [(MCT.NodeMCT.get_score_of_child(self, node, is_initiator), node) for node in children]
            best_child = max(score_node_pairs, key=lambda x: x[0], default=(None, None))[1]
            return best_child

        @classmethod
        def get_score_of_child(cls, parent_node, child_node, is_initiator, c=math.sqrt(2)):
            if is_initiator:
                return child_node.record[0] / child_node.record[2] + c * math.sqrt(
                    math.log(parent_node.record[2]) / child_node.record[2])
            else:
                return child_node.record[1] / child_node.record[2] + c * math.sqrt(
                    math.log(parent_node.record[2]) / child_node.record[2])

        @property
        def children(self):
            return self._next[:]

        @property
        def state(self):
            return self._op

        @property
        def record(self):
            return self._val

    @unique
    class Stage(Enum):
        selection = 0
        simulation = 1
        back_propagation = 2

    def __init__(self, sovereignty_upon_draw="none"):
        """
        Create an empty MCT.
        self._root: root of the tree
        self._size: total number of nodes (except root node) in the tree
        """
        self._root = MCT.NodeMCT()
        self._size = 0
        try:
            UltimateTicTacToe(sovereignty_upon_draw=sovereignty_upon_draw)
        except Exception as e:
            raise e
        else:
            self._sovereignty_upon_draw = sovereignty_upon_draw

    @classmethod
    def offline_learning(cls, tree, epoch_num=1000):
        """
        Self-training without opponent.
        :param tree: An MCT object. The tree to be trained.
        :param epoch_num: Total number of game rounds to be simulated.
        :return:
        """
        n_exploration, n_exploitation = 0, 0
        for epoch in range(1, epoch_num+1):
            print("Training epoch: {}".format(epoch))
            terminal = False
            current_node = tree._root
            initiator, opponent = "X", "O"
            current_side = initiator
            board = UltimateTicTacToe(sovereignty_upon_draw=tree._sovereignty_upon_draw)
            stage = MCT.Stage.selection
            path = [current_node]
            while not terminal:
                print(board)
                # Selection
                if stage == MCT.Stage.selection:
                    explored_actions = [node.state for node in current_node.children]
                    valid_actions = board.valid_actions
                    actions_to_be_explored = list(set(valid_actions) - set(explored_actions))
                    if len(list(set(explored_actions)-set(valid_actions))) > 0:
                        raise RuntimeError("Explored invalid actions!")
                    if len(actions_to_be_explored) > 0:
                        action = random.choice(actions_to_be_explored)
                        stage = MCT.Stage.simulation
                        current_node = tree._add_node(current_node, action)
                        path.append(current_node)
                        n_exploration += 1
                    else:
                        current_node = current_node.get_best_child(current_side == initiator)
                        path.append(current_node)
                        action = current_node.state
                        n_exploitation += 1
                    print("Taking action {}".format(action))
                    terminal = board.take(*action, current_side)

                elif stage == MCT.Stage.simulation:
                    action = board.random_action
                    current_node = tree._add_node(current_node, action)
                    print("Taking action {}".format(action))
                    terminal = board.take(*action, current_side)
                    path.append(current_node)
                    n_exploration += 1

                current_side = "X" if current_side == "O" else "O"
            # back-propagation
            if board.occupancy == initiator:
                result = MCT.NodeMCT.Result.the_initiator_wins
            elif board.occupancy == opponent:
                result = MCT.NodeMCT.Result.the_initiator_loses
            elif board.occupancy == "draw":
                result = MCT.NodeMCT.Result.draw
            else:
                raise ValueError("Failed to get expected result.")
            for node in path:
                node.update(result)
        return n_exploitation, n_exploration

    @classmethod
    def online_learning(cls, model, input_stream, side="X", as_initiator=True, num_eval_for_each_step=1000):
        """
        Online learning with MCTS.
        :param side: "X" or "O"
        :param as_initiator: bool. True if the input serves as the initiator the game.
        :param num_eval_for_each_step: Number action evaluation for each step.
        :return: (action_of_AI, action_info)
        action_of_AI: the coordinate of the slot that the AI just took. None if the game is terminated.
        action_info: Some information for describing the action for debug purposes.
        """
        if side in ("X", "O"):
            opponent = "X" if side == "O" else "O"
        else:
            raise ValueError('Invalid input for parameter "side", expected "X" or "O", got {}. '.format(side))
        if num_eval_for_each_step < 0:
            raise ValueError('Parameter "num_eval_for_each_step" must be greater than 0. ')
        initiator_side, defender_side = (side, opponent) if as_initiator else (opponent, side)
        current_side = initiator_side
        terminal = False
        board = UltimateTicTacToe(sovereignty_upon_draw=model.rule_set["sovereignty_upon_draw"])
        current_node = model._root
        node_path = [model._root]
        while not terminal:
            if current_side == side:  # The input's turn
                while True:
                    try:
                        action = next(input_stream)
                    except StopIteration:
                        raise RuntimeError("Can't receive action from player. ")
                    if action not in board.valid_actions:
                        print("Invalid action. Please try again.")
                        continue
                    else:
                        terminal = board.take(*action, current_side)
                        break
                for node in current_node.children:
                    if node.state == action:
                        current_node = node
                    else:
                        current_node = model._add_node(current_node, action)
                    node_path.append(current_node)
            else:  # The model's turn
                # To be fixed
                for testEpoch in range(num_eval_for_each_step):
                    test_terminal = False
                    test_current_node = current_node
                    test_board = copy.deepcopy(board)
                    test_current_side = current_side
                    test_node_path = []
                    # Selection
                    while not test_terminal:
                        valid_actions = test_board.valid_actions
                        explored_actions = [node.state for node in test_current_node.children]
                        actions_to_be_explored = list(set(valid_actions)-set(explored_actions))
                        if len(actions_to_be_explored) > 0:
                            action = random.choice(actions_to_be_explored)
                            test_current_node = model._add_node(test_current_node, action)
                            test_node_path.append(test_current_node)
                            test_terminal = test_board.take(*action, test_current_side)
                            test_current_side = test_board.next_side
                            break
                        else:
                            test_current_node = test_current_node.get_best_child(test_current_side == initiator_side)
                            test_node_path.append(test_current_node)
                            action = test_current_node.state
                            test_terminal = test_board.take(*action, test_current_side)
                            test_current_side = test_board.next_side

                    # Simulation
                    while not test_terminal:
                        action = test_board.random_action
                        test_current_node = model._add_node(test_current_node, action)
                        test_node_path.append(test_current_node)
                        test_terminal = test_board.take(*action, test_current_side)
                        test_current_side = test_board.next_side
                    # Back-propagation
                    occupancy = test_board.occupancy
                    if occupancy == "draw":
                        result = MCT.NodeMCT.Result.draw
                    elif occupancy == initiator_side:
                        result = MCT.NodeMCT.Result.the_initiator_wins
                    elif occupancy == defender_side:
                        result = MCT.NodeMCT.Result.the_initiator_loses
                    else:
                        raise ValueError("Invalid occupancy when the game terminates.")
                    for node in node_path+test_node_path:
                        node.update(result)
                is_initiator = current_side == initiator_side
                best_node = current_node.get_best_child(is_initiator)
                score = MCT.NodeMCT.get_score_of_child(current_node, best_node, is_initiator)
                current_node = best_node  # Update node reference
                action = current_node.state
                node_path.append(current_node)
                terminal = board.take(*action, opponent)
                yield action, {"board": copy.deepcopy(board), "score": score, "log": current_node.record}
            current_side = "X" if current_side == "O" else "O"

        # Final back-propagation
        occupancy = board.occupancy
        if occupancy == "draw":
            result = MCT.NodeMCT.Result.draw
        elif occupancy == initiator_side:
            result = MCT.NodeMCT.Result.the_initiator_wins
        elif occupancy == defender_side:
            result = MCT.NodeMCT.Result.the_initiator_loses
        else:
            raise ValueError("Invalid occupancy when the game terminates.")
        for node in node_path:
            node.update(result)
        #  Yield last result if it's the input who ended the game
        if current_side == opponent:
            yield None, {"board": board, "score": score, "log": current_node.record}

    def _add_node(self, parent, action):
        node = parent.add_child(action)
        self._size += 1
        return node

    @property
    def size(self):
        return self._size

    @property
    def rule_set(self):
        return {"sovereignty_upon_draw": self._sovereignty_upon_draw}

    @property
    def knowledge_brief(self):
        return self._root.record

    @classmethod
    def save_model(cls, tree, model_path):
        if type(tree) != MCT:
            raise TypeError("Invalid type of tree")
        try:
            with open(os.path.normpath(model_path), "wb") as fileObj:
                pickle.dump(tree, fileObj)
        except IOError as e:
            print(e)

    @classmethod
    def load_model(cls, model_path):
        try:
            with open(os.path.normpath(model_path), "rb") as fileObj:
                return pickle.load(fileObj)
        except IOError as e:
            print(e)
            return None

    def __repr__(self):
        return "Knowledge base size: {}\nOverall: {}\n".format(self.size, self._root.record)


if __name__ == "__main__":
    path1 = r"../model/normal.pkl"
    path2 = r"../model/bizarre.pkl"
    tree1 = MCT.load_model(path1)
    if tree1 is None:
        tree1 = MCT(sovereignty_upon_draw="none")
    print(tree1)
    result = MCT.offline_learning(tree1, 0)
    print("#Exploitation:{}\n#Exploration:{}\n".format(*result))
    print(tree1)
    MCT.save_model(tree1, path1)
    tree2 = MCT.load_model(path2)
    if tree2 is None:
        tree2 = MCT(sovereignty_upon_draw="both")
    print(tree2)
    result = MCT.offline_learning(tree2, 0)
    print("#Exploitation:{}\n#Exploration:{}\n".format(*result))
    print(tree2)
    MCT.save_model(tree2, path2)

