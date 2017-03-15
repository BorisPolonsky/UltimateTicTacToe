import copy
class BigTicTacToe():
    class TicTacToe():
        def __init__(self, **kwargs):
            self.slots=[[0]*3]*3
            self.occupancy=None
            self.full=False
            return super().__init__(**kwargs)

        def Take(self,row,column,side):
            """
            rtype: bool. True if a side wins the TicTacToe
            """
            row-=1
            column-=1
            if not (row in [0,1,2] and column in [0,1,2]):
                raise ValueError("Invalid coordinate.")
            if (self.slots[row][column]==" "):
                if side=="O" or side=="X":
                    self.slots[row][column]=side
                    #Searh row, column and diagonal
                    if (self.slots[row][0]==self.slots[row][1] and self.slots[row][1]==self.slots[row][2])\
                    or (self.slots[0][column]==self.slots[1][column] and self.slots[1][column]==self.slots[2][column])\
                    or (row==column and self.slots[0][0]==self.slots[1][1] and self.slots[1][1]==self.slots[2][2])\
                    or (row+column==2 and self.slots[0][2]==self.slots[1][1] and self.slots[1][1]==self.slots[2][0]):
                        self.occupancy=side
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

    def __init__(self, **kwargs):
        self.blocks=[[TicTacToe()]*3]*3
        self.occupancy=[[None]*3]*3
        self.nextSide="O"
        self.nextBlock=None
        return super().__init__(**kwargs)
        def Take(self,**kwargs):
            """
            rtype: bool. True if a side wins the TicTacToe
            """
            kwargs["row_block"]-=1
            kwargs["column_block"]-=1
            kwargs["row_slot"]-=1
            kwargs["column_slot"]-=1
            if not (kwargs["row_block"] in [0,1,2] and kwargs["column_block"] in [0,1,2]):
                raise ValueError("Invalid coordinate.")
            if (self.blocks[kwargs["row_block"]][kwargs["column_block"]].occupancy==None):
                if side=="O" or side=="X":
                    self.slots[row][column]=side
                    #Searh row, column and diagonal
                    if (self.blocks[kwargs["row_block"]][0].occupancy==self.blocks[kwargs["row_block"]][1].occupancy and self.blocks[kwargs["row_block"]][1].occupancy==self.blocks[kwargs["row_block"]][2].occupancy)\
                    or (self.blocks[0][kwargs["column_block"]].occupancy==self.blocks[1][kwargs["column_block"]].occupancy and self.blocks[1][kwargs["column_block"]].occupancy==self.blocks[2][kwargs["column_block"]].occupancy)\
                    or (kwargs["row_block"]==kwargs["column_block"] and self.blocks[0][0].occupancy==self.blocks[1][1].occupancy and self.blocks[1][1].occupancy==self.blocks[2][2].occupancy)\
                    or (kwargs["row_block"]+kwargs["column_block"]==2 and self.blocks[0][2].occupancy==self.blocks[1][1].occupancy and self.blocks[1][1].occupancy==self.blocks[2][0].occupancy):
                        self.occupancy=side
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The block is already occupied!")