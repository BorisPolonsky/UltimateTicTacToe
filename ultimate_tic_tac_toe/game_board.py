import random
from enum import IntEnum
import numpy as np
from typing import Tuple, Optional, List


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
        def __init__(self, slots: np.ndarray, occupancy: BoardState, num_filled_slots: int, **kwargs):
            assert slots.shape == (3, 3)
            self._slots: np.ndarray = slots
            self._occupancy: BoardState = occupancy
            self._num_filled_slots: int = num_filled_slots
            super().__init__(**kwargs)

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
                        self._occupancy = side
                        return True
                    if self._num_filled_slots == 9:
                        self._occupancy = BoardState.DRAW
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

        @property
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

    def __init__(self,
                 slots: np.ndarray,
                 blocks: Tuple[Tuple[TicTacToe]],
                 next_player_side,
                 next_block_coordinate: Optional[Tuple[int, int]],
                 num_filled_blocks: int,
                 occupancy,
                 sovereignty_upon_draw="none"):
        self._slots: np.ndarray = slots
        self._blocks: Tuple[Tuple[UltimateTicTacToe.TicTacToe]] = blocks
        self._next_player_side: Optional[BoardState] = next_player_side
        self._next_block_coordinate: Tuple[int, int] = next_block_coordinate
        self._num_filled_blocks = num_filled_blocks  # Total number of blocks that has been occupied.
        self._occupancy = occupancy
        if sovereignty_upon_draw in ("none", "both"):
            self._sovereignty_upon_draw = sovereignty_upon_draw
        else:
            raise ValueError('Invalid rule set. "sovereignty_upon_draw" should be either "both" or "none".')
        super().__init__()

    @classmethod
    def create_initial_board(cls, **kwargs):
        """
        :param kwargs:
        """
        def split_board_into_blocks(slots):
            blocks: List[Tuple[UltimateTicTacToe.TicTacToe]] = []
            for row in range(0, 9, 3):
                block_buff = []
                for column in range(0, 9, 3):
                    block_buff.append(UltimateTicTacToe.TicTacToe(slots=slots[row:row + 3, column:column + 3], occupancy=BoardState.UNOCCUPIED, num_filled_slots=0))
                blocks.append(tuple(block_buff))
            blocks: Tuple[Tuple[UltimateTicTacToe.TicTacToe]] = tuple(blocks)
            return blocks

        slots = np.zeros([9, 9], dtype=np.int32)
        blocks = split_board_into_blocks(slots)
        next_block_coordinate = None
        num_filled_blocks = 0  # Total number of blocks that has been occupied.
        occupancy = None
        if not ("sovereignty_upon_draw" in kwargs):
            sovereignty_upon_draw = "none"
        elif kwargs["sovereignty_upon_draw"] in ("none", "both"):
            sovereignty_upon_draw = kwargs["sovereignty_upon_draw"]
        else:
            raise ValueError('Invalid rule set. "sovereignty_upon_draw" should be either "both" or "none".')
        if "initiator" in kwargs:
            raise ValueError('"initiator" in args, yet this parameter has been removed.')
        else:
            next_player_side = SlotState.PLAYER1
        board = UltimateTicTacToe(slots=slots,
                                  blocks=blocks,
                                  next_player_side=next_player_side,
                                  next_block_coordinate=next_block_coordinate,
                                  num_filled_blocks=num_filled_blocks,
                                  occupancy=occupancy,
                                  sovereignty_upon_draw=sovereignty_upon_draw)
        return board

    def copy(self):
        # Make a copy of slots
        new_slots = self._slots.copy()
        # Create blocks according given slots
        new_blocks = []
        for row in range(3):
            block_buff = []
            for col in range(3):
                current_block = self._blocks[row][col]
                current_block_slots = new_slots[3 * row:3 * (row + 1), 3 * col: 3 * (col + 1)]
                new_block = self.__class__.TicTacToe(slots=current_block_slots,
                                                     occupancy=current_block._occupancy,
                                                     num_filled_slots=current_block._num_filled_slots)
                block_buff.append(new_block)
            new_blocks.append(tuple(block_buff))
        new_blocks = tuple(new_blocks)
        return UltimateTicTacToe(slots=new_slots,
                                 blocks=new_blocks,
                                 next_player_side=self._next_player_side,
                                 next_block_coordinate=self._next_block_coordinate,
                                 num_filled_blocks=self._num_filled_blocks,
                                 occupancy=self._occupancy,
                                 sovereignty_upon_draw=self._sovereignty_upon_draw)

    def take(self, row_block: int, column_block: int, row_slot: int, column_slot: int, side):
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
        if self._blocks[row_block][column_block].occupancy == BoardState.UNOCCUPIED:
            if (side == SlotState.PLAYER1 or side == SlotState.PLAYER2) and (self._next_player_side == side):
                terminal = self._blocks[row_block][column_block].take(row_slot, column_slot, side)
                self._next_player_side = SlotState.PLAYER1 if side == SlotState.PLAYER2 else SlotState.PLAYER2
                self._next_block_coordinate = None if self._blocks[row_slot][column_slot].occupancy != BoardState.UNOCCUPIED else (
                    row_slot, column_slot)
                if terminal:
                    self._num_filled_blocks += 1
                    horizontal = (self._blocks[row_block][0].occupancy, self._blocks[row_block][1].occupancy,
                                  self._blocks[row_block][2].occupancy)
                    vertical = (self._blocks[0][column_block].occupancy, self._blocks[1][column_block].occupancy,
                                self._blocks[2][column_block].occupancy)
                    diagonal1 = (
                        self._blocks[0][0].occupancy, self._blocks[1][1].occupancy, self._blocks[2][2].occupancy)
                    diagonal2 = (
                        self._blocks[2][0].occupancy, self._blocks[1][1].occupancy, self._blocks[0][2].occupancy)
                    if self._sovereignty_upon_draw == "both":
                        horizontal = list(map(lambda x: x if x != BoardState.DRAW else side, horizontal))
                        vertical = list(map(lambda x: x if x != BoardState.DRAW else side, vertical))
                        diagonal1 = list(map(lambda x: x if x != BoardState.DRAW else side, diagonal1))
                        diagonal2 = list(map(lambda x: x if x != BoardState.DRAW else side, diagonal2))
                    # Search row, column and diagonal
                    if (side == horizontal[0] == horizontal[1] == horizontal[2]) \
                        or (side == vertical[0] == vertical[1] == vertical[2]) \
                        or (row_block == column_block and side == diagonal1[0] == diagonal1[1] == diagonal1[2]) \
                            or (row_block + column_block == 2 and side == diagonal2[0] == diagonal2[1] == diagonal2[2]):
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
        :return: Return a list of actions [(block_row, block_column,slot_row,slot_column),...]
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
                SlotState.PLAYER1.value: " {} ".format(token1),
                SlotState.PLAYER2.value: " {} ".format(token2),
                SlotState.UNOCCUPIED.value: "   "
            }[state_id]

        lines = []
        for row_idx, row in enumerate(self._slots):
            if row_idx in (3, 6):
                lines.append("=====================================")
            elif 0 < row_idx < 9:
                lines.append("-------------------------------------")
            line = list(map(state2token, row))
            line_buff = []
            for col_idx in range(0, 9, 3):
                line_buff.append("|".join(line[col_idx:col_idx+3]))
            line = "||".join(line_buff)
            lines.append(line)

        return "\n".join(lines)

