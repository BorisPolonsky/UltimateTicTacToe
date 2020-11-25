import pickle
import os
import math
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe, BoardState, SlotState
import random
from enum import Enum, unique
from typing import Dict, Tuple, Iterable, Optional, Union
import numpy as np


class MCT:
    class Node:
        def __init__(self, *args, **kwargs):
            pass

    class DummyNode(Node):
        def __init__(self):
            self._parent = None
            self.children = {}
            pass

        def update(self, *args, **kwargs):
            pass

        def _update_child_game_log(self, *args, **kwargs):
            pass

        def add_child(self, *args, **kwargs):
            action_id = 0
            node = MCT.ActualNode(op=action_id, parent=self)
            self.children[action_id] = node
            return node

    class ActualNode(Node):
        def __init__(self, op: Optional[Union[int, Tuple[int, int, int, int]]] = None, parent=None):
            """
            Initialization of a node.
            self._op: the operation from last state to reach this node.
            None for root, (row_block, row_column, row_slot, column_slot) for the rest.
            self._val: a tuple of (#game in which the initiator wins, #game in which the initiator loses, #total games)
            #draw=self._val[2]-self._val[0]-self._val[1]
            self._next: a list of children of the node.
            """
            if isinstance(op, tuple):
                op = MCT._action2id(op)
            if (op is None) or (isinstance(op, int) and (0 <= op < 81)):
                self._action_id = op
            else:
                raise ValueError("Invalid op: {}".format(op))
            self.children: Dict[int, MCT.ActualNode] = {}
            self._parent: Optional[MCT.ActualNode] = parent
            self._child_game_log = np.zeros([3, 81])  # dim_0: [initator_wins, initiator_loses, draw], dim_1: [action_id0, action_id1, ..., action_id80]

        def add_child(self, action_id: int):
            if not isinstance(action_id, int):
                print("{}".format(action_id))
            node = MCT.ActualNode(action_id, parent=self)
            assert action_id not in self.children
            self.children[action_id] = node
            return node

        def _update_child_game_log(self, action_id, result):
            if result == MCT.Result.THE_INITIATOR_WINS:
                log_to_update = self._child_game_log[0]
            elif result == MCT.Result.THE_INITIATOR_LOSES:
                log_to_update = self._child_game_log[1]
            elif result == MCT.Result.DRAW:
                log_to_update = self._child_game_log[2]
            else:
                raise ValueError("Invalid result.")
            log_to_update[action_id] += 1

        def update(self, result):
            action_id = self._action_id
            self._parent._update_child_game_log(action_id, result)

        def get_best_child(self, is_initiator, c=math.sqrt(2)):
            if c < 0:
                raise ValueError('Parameter c must be greater or equal to 0. ')
            if not self.children:
                raise ValueError("There's no child for current node.")
            vals = self.get_children_values(is_initiator=is_initiator, c=c)
            best_action = max(self.children.keys(), key=lambda k: vals[k])
            best_child = self.children[best_action]
            return best_child

        @property
        def total_visit(self):
            return np.sum(self._child_game_log)

        def get_children_values(self, is_initiator, c=math.sqrt(2)):
            if is_initiator:
                n_win, n_lose, n_draw = self._child_game_log
            else:
                n_lose, n_win, n_draw = self._child_game_log
            child_total = np.sum(self._child_game_log, axis=0)
            child_total_plus_one = child_total + 1
            self_total = np.sum(child_total)
            child_q_vals = n_win / child_total_plus_one
            child_u_vals = c * np.sqrt(np.log(self_total + 1) / (child_total_plus_one))
            return child_q_vals + child_u_vals

        @property
        def action_id(self):
            return self._action_id

        @property
        def record(self):
            return np.sum(self._child_game_log, axis=1)

    @unique
    class Stage(Enum):
        SELECTION = 0
        SIMULATION = 1
        BACK_PROPAGATION = 2

    @unique
    class Result(Enum):
        THE_INITIATOR_WINS = 0
        THE_INITIATOR_LOSES = 1
        DRAW = 2
    def __init__(self, sovereignty_upon_draw="none"):
        """
        Create an empty MCT.
        self._root: root of the tree
        self._size: total number of nodes (except root node) in the tree
        """
        root = MCT.DummyNode()
        root = root.add_child(op=0)
        self._root = root
        self._size = 0  # Not taking dummy node into account
        try:
            UltimateTicTacToe.create_initial_board(sovereignty_upon_draw=sovereignty_upon_draw)
        except Exception as e:
            raise e
        else:
            self._sovereignty_upon_draw = sovereignty_upon_draw

    @classmethod
    def offline_learning(cls, tree, epoch_num=1000, verbose=0):
        """
        Self-training without opponent.
        :param tree: An MCT object. The tree to be trained.
        :param epoch_num: Total number of game rounds to be simulated.
        :param verbose: A parameter for controlling verbosity level of console output.
        :return: n_exploitation, n_exploration
        """
        n_exploration, n_exploitation = 0, 0
        initiator, opponent = SlotState.PLAYER1, SlotState.PLAYER2
        for epoch in range(1, epoch_num + 1):
            if verbose >= 1:
                print("Training epoch {} in a total of {}: ".format(epoch, epoch_num))
            terminal = False
            current_node = tree._root
            current_side = initiator
            board = UltimateTicTacToe.create_initial_board(sovereignty_upon_draw=tree._sovereignty_upon_draw)
            stage = MCT.Stage.SELECTION
            path = [current_node]
            while not terminal:
                if verbose >= 2:
                    print(board)
                # Selection
                if stage == MCT.Stage.SELECTION:
                    explored_actions = set(current_node.children.keys())
                    valid_actions = set(map(lambda action_tuple: cls._action2id(action_tuple), board.valid_actions))
                    actions_to_be_explored = list(set(valid_actions) - set(explored_actions))
                    if len(list(set(explored_actions)-set(valid_actions))) > 0:
                        raise RuntimeError("Explored invalid actions!")
                    if len(actions_to_be_explored) > 0:
                        action_id = random.choice(actions_to_be_explored)
                        stage = MCT.Stage.SIMULATION
                        current_node = tree._add_node(current_node, action_id)
                        path.append(current_node)
                        n_exploration += 1
                    else:
                        current_node = current_node.get_best_child(current_side == initiator)
                        path.append(current_node)
                        action_id = current_node.action_id
                        n_exploitation += 1
                    action_tuple = cls._id2action(action_id)
                    if verbose >= 2:
                        print("Taking action {}.".format(action_tuple))
                    terminal = board.take(*action_tuple, current_side)

                elif stage == MCT.Stage.SIMULATION:
                    action_id = board.random_action()
                    current_node = tree._add_node(current_node, action_id)
                    if verbose >= 2:
                        print("Taking action {}.".format(action_id))
                    terminal = board.take(*action_id, current_side)
                    path.append(current_node)
                    n_exploration += 1

                current_side = SlotState.PLAYER2 if current_side == SlotState.PLAYER1 else SlotState.PLAYER1
            # back-propagation
            if board.occupancy == BoardState.OCCUPIED_BY_PLAYER1:
                result = MCT.Result.THE_INITIATOR_WINS
            elif board.occupancy == BoardState.OCCUPIED_BY_PLAYER2:
                result = MCT.Result.THE_INITIATOR_LOSES
            elif board.occupancy == BoardState.DRAW:
                result = MCT.Result.DRAW
            else:
                raise ValueError("Failed to get expected result.")
            for node in path:
                node.update(result)
        return n_exploitation, n_exploration

    @classmethod
    def online_learning(cls, model,
                        input_stream: Iterable[Tuple[int, int, int, int]],
                        as_initiator=True,
                        num_eval_for_each_step=1000):
        """
        Online learning with MCTS.
        :param model
        :param input_stream
        :param as_initiator: bool. True if the input serves as the initiator the game.
        :param num_eval_for_each_step: Number action evaluation for each step.
        :return: (action_of_AI, action_info)
        action_of_AI: the coordinate of the slot that the AI just took. None if the game is terminated.
        action_info: Some information for describing the action for debug purposes.
        """
        initiator, defender = SlotState.PLAYER1, SlotState.PLAYER2
        user_side, ai_side = (initiator, defender) if as_initiator else (defender, initiator)
        if num_eval_for_each_step < 0:
            raise ValueError('Parameter "num_eval_for_each_step" must be greater than 0. ')
        current_side = initiator
        terminal = False
        board = UltimateTicTacToe.create_initial_board(sovereignty_upon_draw=model.rule_set["sovereignty_upon_draw"])
        current_node = model._root
        node_path = [model._root]
        while not terminal:
            if current_side == user_side:  # The user's turn
                while True:
                    try:
                        action = next(input_stream)
                    except StopIteration:
                        raise RuntimeError("Can't receive action from player.")
                    if action not in board.valid_actions:
                        print("Invalid action. Please try again.")
                        continue
                    else:
                        terminal = board.take(*action, current_side)
                        break
                #for action_id, node in current_node.children.items():
                action_id = cls._action2id(action)
                if action_id in current_node.children:
                    current_node = current_node[action_id]
                else:
                    current_node = model._add_node(current_node, action_id)
                node_path.append(current_node)
            else:  # The model's turn
                # To be fixed
                for test_epoch in range(num_eval_for_each_step):
                    test_terminal = False
                    test_current_node = current_node
                    test_board = board.copy()
                    test_current_side = current_side
                    test_node_path = []
                    # Selection
                    while not test_terminal:
                        valid_actions = test_board.valid_actions
                        explored_actions = {MCT._id2action(action_id) for action_id in test_current_node.children}
                        actions_to_be_explored = list(set(valid_actions) - explored_actions)
                        if len(actions_to_be_explored) > 0:
                            action = random.choice(actions_to_be_explored)
                            test_current_node = model._add_node(test_current_node, action)
                            test_node_path.append(test_current_node)
                            test_terminal = test_board.take(*action, test_current_side)
                            test_current_side = test_board.next_side
                            break
                        else:
                            test_current_node = test_current_node.get_best_child(test_current_side == initiator)
                            test_node_path.append(test_current_node)
                            action = MCT._id2action(test_current_node.action_id)
                            test_terminal = test_board.take(*action, test_current_side)
                            test_current_side = test_board.next_side

                    # Simulation
                    while not test_terminal:
                        action = test_board.random_action()
                        test_current_node = model._add_node(test_current_node, action)
                        test_node_path.append(test_current_node)
                        test_terminal = test_board.take(*action, test_current_side)
                        test_current_side = test_board.next_side
                    # Back-propagation
                    occupancy = test_board.occupancy
                    if occupancy == BoardState.DRAW:
                        result = MCT.Result.DRAW
                    elif occupancy == BoardState.OCCUPIED_BY_PLAYER1:
                        result = MCT.Result.THE_INITIATOR_WINS
                    elif occupancy == BoardState.OCCUPIED_BY_PLAYER2:
                        result = MCT.Result.THE_INITIATOR_LOSES
                    else:
                        raise ValueError("Invalid occupancy when the game terminates.")
                    for node in node_path + test_node_path:
                        node.update(result)
                is_initiator = current_side == initiator
                best_node = current_node.get_best_child(is_initiator)
                score = current_node.get_children_values(is_initiator)[best_node.action_id]  # to be optimized: Duplicated value calculation
                current_node = best_node  # Update node reference
                action = MCT._id2action(current_node.action_id)
                node_path.append(current_node)
                terminal = board.take(*action, ai_side)
                yield action, {"board": board.copy(), "score": score, "log": current_node.record}
            current_side = SlotState.PLAYER1 if current_side == SlotState.PLAYER2 else SlotState.PLAYER2

        # Final back-propagation
        occupancy = board.occupancy
        if occupancy == BoardState.DRAW:
            result = MCT.Result.DRAW
        elif occupancy == BoardState.OCCUPIED_BY_PLAYER1:
            result = MCT.Result.THE_INITIATOR_WINS
        elif occupancy == BoardState.OCCUPIED_BY_PLAYER2:
            result = MCT.Result.THE_INITIATOR_LOSES
        else:
            raise ValueError("Invalid occupancy when the game terminates.")
        for node in node_path:
            node.update(result)
        #  Yield last result if it's the input who ended the game
        if current_side == user_side:
            yield None, {"board": board, "score": score, "log": current_node.record}

    def _add_node(self, parent, action: Union[int, Tuple[int, int, int, int]]):
        action_id = self._action2id(action) if isinstance(action, tuple) else action
        node = parent.add_child(action_id)
        self._size += 1
        return node

    @classmethod
    def _action2id(cls, op: Tuple[int, int, int, int]):
        if isinstance(op, tuple) and len(op) == 4:
            for val in op:
                if val not in range(3):
                    raise ValueError("Each entry of the state must be an integer from 1-3")
            row_block, column_block, row_slot, column_slot = op
            row = row_block * 3 + row_slot
            column = column_block * 3 + column_slot
            idx = row * 9 + column
            return idx
        else:
            raise TypeError(
                'Parameter "op" must be either None or a tuple of (row_block, column_block, row_slot, column_slot)')

    @classmethod
    def _id2action(cls, idx: int) -> Tuple[int, int, int, int]:
        row, column = divmod(idx, 9)
        row_block, row_slot = divmod(row, 3)
        column_block, column_slot = divmod(column, 3)
        return row_block, column_block, row_slot, column_slot

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
            with open(os.path.normpath(model_path), "wb") as f:
                pickle.dump(tree, f)
        except IOError as e:
            print(e)

    @classmethod
    def load_model(cls, model_path):
        try:
            with open(os.path.normpath(model_path), "rb") as f:
                return pickle.load(f)
        except IOError as e:
            print(e)
            return None

    #def __repr__(self):
        #return "Knowledge base size: {}\nOverall: {}\n".format(self.size, self._root.record)


if __name__ == "__main__":
    path1 = r"../model/normal.pkl"
    path2 = r"../model/bizarre.pkl"
    tree1 = MCT.load_model(path1)
    if tree1 is None:
        tree1 = MCT(sovereignty_upon_draw="none")
    print(tree1)
    result = MCT.offline_learning(tree1, 1000)
    print("#Exploitation:{}\n#Exploration:{}\n".format(*result))
    print(tree1)
    MCT.save_model(tree1, path1)
    tree2 = MCT.load_model(path2)
    if tree2 is None:
        tree2 = MCT(sovereignty_upon_draw="both")
    print(tree2)
    result = MCT.offline_learning(tree2, 1000)
    print("#Exploitation:{}\n#Exploration:{}\n".format(*result))
    print(tree2)
    MCT.save_model(tree2, path2)

