import random
from enum import IntEnum
import numpy as np


class SlotState(IntEnum):
    UNOCCUPIED = 0  # Empty slot
    PLAYER1 = 1  # Occupied by the initiator
    PLAYER2 = 2  # Occupied by the other player


class BoardState(IntEnum):
    UNOCCUPIED = 0  # Non-terminal
    OCCUPIED_BY_PLAYER1 = 1  # Terminal
    OCCUPIED_BY_PLAYER2 = 2  # Terminal
    DRAW = 3  # Terminal


class UltimateTicTacToe:
    class TicTacToe:
        def __init__(self, slots, **kwargs):
            assert slots.shape == (3, 3)
            self._slots = slots
            self._occupancy = BoardState.UNOCCUPIED
            self._num_filled_slots = 0
            return super().__init__(**kwargs)

        def take(self, row: int, column: int, side: int):
            """
            Take a slot.
            :param row: 0, 1 or 2
            :param column: 0, 1 or 2
            :param side: 1 or 2
            :return: bool. Return True if:
            1) A side wins the TicTacToe or,
            2) draw.
            """
            if not (row in range(3) and column in range(3)):
                raise ValueError("Invalid coordinate.")
            if self._slots[row, column] == SlotState.UNOCCUPIED:
                if side == SlotState.PLAYER1 or side == SlotState.PLAYER2:
                    self._num_filled_slots += 1
                    self._slots[row, column] = int(side)
                    # Search row, column and diagonals
                    if (self._slots[row, 0] == self._slots[row, 1] == self._slots[row, 2]) \
                        or (self._slots[0, column] == self._slots[1, column] == self._slots[2, column]) \
                        or (row == column and self._slots[0, 0] == self._slots[1, 1] == self._slots[2, 2]) \
                        or (row + column == 2 and self._slots[0, 2] == self._slots[1, 1] == self._slots[2, 0]):
                        self._occupancy = BoardState.UNOCCUPIED
                        return True
                    if self._num_filled_slots == 9:
                        self._occupancy = BoardState.DRAW
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

        def occupancy(self):
            return self._occupancy

        @property
        def valid_actions(self):
            """
            Return a list of actions : [(slotRow,slotColumn),...]
            Each index starts with 0.
            :return: a list of actions : [(slotRow,slotColumn),...]
            """
            if self._occupancy != BoardState.UNOCCUPIED:
                return []
            actions = []
            for row in range(3):
                for column in range(3):
                    if self._slots[row, column] == SlotState.UNOCCUPIED:
                        actions.append((row, column))
            return actions

    def __init__(self, **kwargs):
        """
        :param kwargs:
        """
        self._slots = np.zeros([9, 9], dtype=np.int32)
        self._blocks = self.__class__._split_board_into_blocks(self._slots)
        self._next_player_side = SlotState.PLAYER1
        self._next_block_coordinate = None
        self._num_filled_blocks = 0  # Total number of blocks that has been occupied.
        self._occupancy = None
        if not ("sovereignty_upon_draw" in kwargs):
            self._sovereignty_upon_draw = "none"
        elif kwargs["sovereignty_upon_draw"] in ("none", "both"):
            self._sovereignty_upon_draw = kwargs["sovereignty_upon_draw"]
        else:
            raise ValueError('Invalid rule set. "sovereignty_upon_draw" should be either "both" or "none".')

        if "initiator" in kwargs:
            print("initiator in args, yet this parameter will be removed.")
        else:
            self._next_player_side = SlotState.PLAYER1
        return super().__init__()

    @classmethod
    def _split_board_into_blocks(cls, slots):
        blocks = []
        for row in range(0, 9, 3):
            blocks.append([])
            for column in range(0, 9, 3):
                blocks[-1].append(UltimateTicTacToe.TicTacToe(slots[row:row + 3, column:column + 3]))
        return blocks

    def take(self, row_block, column_block, row_slot, column_slot, side):
        """

        :param row_block:
        :param column_block:
        :param row_slot:
        :param column_slot:
        :param side:
        :return: bool. Return True if draw or a side wins the UltimateTicTacToe. Return False if the game is unfinished.
        """
        if not (row_block in range(3) and column_block in range(3)):
            raise ValueError("Invalid coordinate. ")
        if (self._next_block_coordinate is not None) and not ((row_block, column_block) == self._next_block_coordinate):
            raise ValueError("Rule violation. Invalid block. ")
        if self._blocks[row_block][column_block].occupancy() == BoardState.UNOCCUPIED:
            if (side == SlotState.PLAYER1 or side == SlotState.PLAYER2) and (self._next_player_side == side):
                terminal = self._blocks[row_block][column_block].take(row_slot, column_slot, side)
                self._next_player_side = SlotState.PLAYER1 if side == SlotState.PLAYER2 else SlotState.PLAYER2
                self._next_block_coordinate = None if self._blocks[row_slot][column_slot].occupancy() is not None else (
                    row_slot, column_slot)
                if terminal:
                    self._num_filled_blocks += 1
                    horizontal = (self._blocks[row_block][0].occupancy(), self._blocks[row_block][1].occupancy(),
                                  self._blocks[row_block][2].occupancy())
                    vertical = (self._blocks[0][column_block].occupancy(), self._blocks[1][column_block].occupancy(),
                                self._blocks[2][column_block].occupancy())
                    diagonal1 = (
                        self._blocks[0][0].occupancy(), self._blocks[1][1].occupancy(), self._blocks[2][2].occupancy())
                    diagonal2 = (
                        self._blocks[2][0].occupancy(), self._blocks[1][1].occupancy(), self._blocks[0][2].occupancy())
                    if self._sovereignty_upon_draw == "both":
                        horizontal = list(map(lambda x: x if x != BoardState.DRAW else side, horizontal))
                        vertical = list(map(lambda x: x if x != BoardState.DRAW else side, vertical))
                        diagonal1 = list(map(lambda x: x if x != BoardState.DRAW else side, diagonal1))
                        diagonal2 = list(map(lambda x: x if x != BoardState.DRAW else side, diagonal2))
                    # Search row, column and diagonal
                    if (horizontal[0] == horizontal[1] and horizontal[1] == horizontal[2]) \
                        or (vertical[0] == vertical[1] and vertical[1] == vertical[2]) \
                        or (row_block == column_block and diagonal1[0] == diagonal1[1] and diagonal1[1] == diagonal1[2]) \
                        or (
                        row_block + column_block == 2 and diagonal2[0] == diagonal2[1] and diagonal2[1] == diagonal2[
                        2]):
                        self._occupancy = side
                        self._next_player_side = None
                        self._next_block_coordinate = None
                        return True
                    elif self._num_filled_blocks == 9:
                        self._occupancy = BoardState.DRAW
                        self._next_player_side = None
                        self._next_block_coordinate = None
                        return True
                    else:
                        return False
                else:
                    return False  # The game continues.
            else:
                raise ValueError("Invalid input for side.")
        else:
            raise ValueError("Invalid block selection. Block occupied.")

    @property
    def valid_actions(self):
        """
        Return valid actions.
        :return: Return a list of actions [(block_row,block_column,slot_row,slot_column),...]
        """

        if self._next_block_coordinate is None:
            actions = []
            if self._occupancy is not None:
                return actions
            for outer_row in range(3):
                for outer_column in range(3):
                    for inner_row, inner_column in self._blocks[outer_row][outer_column].valid_actions:
                        actions.append((outer_row, outer_column, inner_row, inner_column))
        else:
            outer_row, outer_column = self._next_block_coordinate
            block = self._blocks[outer_row][outer_column]
            actions = [(outer_row, outer_column, inner_row, inner_column) for (inner_row, inner_column) in
                       block.valid_actions]
        return actions

    @property
    def random_action(self):
        actions = self.valid_actions
        if len(actions) > 0:
            return random.choice(actions)
        else:
            return None

    @property
    def occupancy(self):
        return self._occupancy

    @property
    def next_side(self):
        return self._next_player_side

    def __repr__(self):
        return self._slots.__repr__()

    def as_str(self, token1="O", token2="X"):
        """

        :param token1: token for initiator
        :param token2: token for the counterpart
        :return:
        """

        def state2token(state_id):
            return {
                SlotState.PLAYER1.value: token1,
                SlotState.PLAYER2.value: token2,
                SlotState.UNOCCUPIED.value: " "
            }[state_id]



        whole_ultimate_tic_tac_toe = []
        for row in self._slots:
            row = " | ".join(list(map(state2token, row)))
            whole_ultimate_tic_tac_toe.append(row)
        return "\n---------------------------------\n".join(whole_ultimate_tic_tac_toe) + "\n"


if __name__ == "__main__":
    board = UltimateTicTacToe(initiator="X")
    action = board.random_action
    while action is not None:
        board.take(*action, board.next_side)
        action = board.random_action
    print(board)
    print(board.state_report())
