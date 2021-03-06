import random


class UltimateTicTacToe:
    class TicTacToe:
        def __init__(self, **kwargs):
            self._slots = [[" "] * 3 for _ in range(3)]
            self._occupancy = None  # "O" , "X" or "draw"
            self._num_filled_slots = 0
            return super().__init__(**kwargs)

        def take(self, row, column, side):
            """
            Take a slot.
            :param row: 0, 1 or 2
            :param column: 0, 1 or 2
            :param side: "X" or "O"
            :return: bool. Return True if:
            1) A side wins the TicTacToe or,
            2) draw.
            """
            if not (row in range(3) and column in range(3)):
                raise ValueError("Invalid coordinate.")
            if self._slots[row][column] == " ":
                if side == "O" or side == "X":
                    self._num_filled_slots += 1
                    self._slots[row][column] = side
                    # Search row, column and diagonals
                    if (self._slots[row][0] == self._slots[row][1] and self._slots[row][1] == self._slots[row][2])\
                            or (self._slots[0][column] == self._slots[1][column] and self._slots[1][column] == self._slots[2][column])\
                            or (row == column and self._slots[0][0] == self._slots[1][1] and self._slots[1][1] == self._slots[2][2])\
                            or (row+column == 2 and self._slots[0][2] == self._slots[1][1] and self._slots[1][1] == self._slots[2][0]):
                        self._occupancy = side
                        return True
                    if self._num_filled_slots == 9:
                        self._occupancy = "draw"
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

        def slot(self):
            return [row[:] for row in self._slots]

        def occupancy(self):
            return self._occupancy

        @property
        def valid_actions(self):
            """
            Return a list of actions : [(slotRow,slotColumn),...]
            Each index starts with 0.
            :return: a list of actions : [(slotRow,slotColumn),...]
            """
            if self._occupancy is not None:
                return []
            # return [(row,column) for column in range(3) for row in range(3) if self._slots[row][column] ==" "]
            actions = []
            for row in range(3):
                for column in range(3):
                    if self._slots[row][column] == " ":
                        actions.append((row, column))
            return actions

    def __init__(self, **kwargs):
        self._blocks = [[UltimateTicTacToe.TicTacToe() for _ in range(3)] for _ in range(3)]
        # self._occupancy = [[None for _ in range(3)] for _ in range(3)]
        self._next_side = None
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
            if kwargs["initiator"] in ("O", "X"):
                self._next_side = kwargs["initiator"]
            else:
                raise ValueError('Parameter "initiator" must be either "X" or "O"')
        else:
            self._next_side = None
        return super().__init__()

    def take(self, row_block, column_block, row_slot, column_slot, side):
        """

        :param row_block:
        :param column_block:
        :param row_slot:
        :param column_slot:
        :param side:
        :return: bool. Return True if draw or a side wins the UltimateTicTacToe. Return False if the game is unfinished.
        """
        row_block -= 1
        column_block -= 1
        row_slot -= 1
        column_slot -= 1
        if not (row_block in range(3) and column_block in range(3)):
            raise ValueError("Invalid coordinate. ")
        if (self._next_block_coordinate is not None) and not ((row_block, column_block) == self._next_block_coordinate):
            raise ValueError("Rule violation. Invalid block. ")
        if self._blocks[row_block][column_block].occupancy() is None:
            if (side == "O" or side == "X") and (self._next_side is None or self._next_side == side):
                terminal = self._blocks[row_block][column_block].take(row_slot, column_slot, side)
                self._next_side = "O" if side == "X" else "X"
                self._next_block_coordinate = None if self._blocks[row_slot][column_slot].occupancy() is not None else (row_slot, column_slot)
                if terminal:
                    self._num_filled_blocks += 1
                    horizontal = (self._blocks[row_block][0].occupancy(), self._blocks[row_block][1].occupancy(), self._blocks[row_block][2].occupancy())
                    vertical = (self._blocks[0][column_block].occupancy(), self._blocks[1][column_block].occupancy(), self._blocks[2][column_block].occupancy())
                    diagonal1 = (self._blocks[0][0].occupancy(), self._blocks[1][1].occupancy(), self._blocks[2][2].occupancy())
                    diagonal2 = (self._blocks[2][0].occupancy(), self._blocks[1][1].occupancy(), self._blocks[0][2].occupancy())
                    if self._sovereignty_upon_draw == "both":
                        horizontal = list(map(lambda x: x if x != "draw" else side, horizontal))
                        vertical = list(map(lambda x: x if x != "draw" else side, vertical))
                        diagonal1 = list(map(lambda x: x if x != "draw" else side, diagonal1))
                        diagonal2 = list(map(lambda x: x if x != "draw" else side, diagonal2))
                    # Search row, column and diagonal
                    if (horizontal[0] == horizontal[1] and horizontal[1] == horizontal[2])\
                            or (vertical[0] == vertical[1] and vertical[1] == vertical[2])\
                            or (row_block == column_block and diagonal1[0] == diagonal1[1] and diagonal1[1] == diagonal1[2])\
                            or (row_block+column_block == 2 and diagonal2[0] == diagonal2[1] and diagonal2[1] == diagonal2[2]):
                        self._occupancy = side
                        self._next_side = None
                        self._next_block_coordinate = None
                        return True
                    elif self._num_filled_blocks == 9:
                        self._occupancy = "draw"
                        self._next_side = None
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
        Each index starts with 1.
        :return: Return a list of actions [(block_row,block_column,slot_row,slot_column),...]
        """

        if self._next_block_coordinate is None:
            actions = []
            if self._occupancy is not None:
                return actions
            for row in range(3):
                for column in range(3):
                    for coordinate in self._blocks[row][column].valid_actions:
                        actions.append((row+1, column+1, coordinate[0]+1, coordinate[1]+1))
        else:
            actions = [(self._next_block_coordinate[0] + 1, self._next_block_coordinate[1] + 1, action[0] + 1, action[1] + 1) for action in self._blocks[self._next_block_coordinate[0]][self._next_block_coordinate[1]].valid_actions]
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
        return self._next_side

    def state_report(self):
        def states():
            for rowBlock in self._blocks:
                for block in rowBlock:
                    for row in block.slot():
                        for column in row:
                            yield column
        state = dict()
        state["state"] = [element for element in states()]
        return state

    def __repr__(self):
        whole_ultimate_tic_tac_toe = []
        for rowBlock in self._blocks:
            row_ultimate_tic_tac_toe_blocks = []
            for rowSlot in range(3):
                row_ultimate_tic_tac_toe=[]
                for columnBlock in range(3):
                    row_ultimate_tic_tac_toe.append(" | ".join(rowBlock[columnBlock].slot()[rowSlot]))
                row_ultimate_tic_tac_toe_blocks.append(" "+"||".join(row_ultimate_tic_tac_toe)+" ")
            whole_ultimate_tic_tac_toe.append("\n---------------------------------\n".join(row_ultimate_tic_tac_toe_blocks))
        return "\n=================================\n".join(whole_ultimate_tic_tac_toe)+"\n"


if __name__ == "__main__":
    board = UltimateTicTacToe(initiator="X")
    action = board.random_action
    while action is not None:
        board.take(*action, board.next_side)
        action = board.random_action
    print(board)
    print(board.state_report())
