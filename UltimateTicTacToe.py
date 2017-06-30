import copy
class UltimateTicTacToe():
    class TicTacToe():
        def __init__(self, **kwargs):
            self.slots=[[" "]*3 for i in range(3)]
            self.occupancy=None#"O" , "X" or "draw"
            self.numFilledSlots=0
            return super().__init__(**kwargs)

        def Take(self,row,column,side):
            """
            rtype: bool. Return True if a side wins the TicTacToe. Return False if draw or not full.  
            """
            if not (row in [0,1,2] and column in [0,1,2]):
                raise ValueError("Invalid coordinate.")
            if (self.slots[row][column]==" "):
                if side=="O" or side=="X":
                    self.numFilledSlots+=1
                    self.slots[row][column]=side
                    #Searh row, column and diagonal
                    if (self.slots[row][0]==self.slots[row][1] and self.slots[row][1]==self.slots[row][2])\
                    or (self.slots[0][column]==self.slots[1][column] and self.slots[1][column]==self.slots[2][column])\
                    or (row==column and self.slots[0][0]==self.slots[1][1] and self.slots[1][1]==self.slots[2][2])\
                    or (row+column==2 and self.slots[0][2]==self.slots[1][1] and self.slots[1][1]==self.slots[2][0]):
                        self.occupancy=side
                        return True
                    if self.numFilledSlots==9:
                        self.occupancy="draw"
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

    def __init__(self, **kwargs):
        self.blocks=[[UltimateTicTacToe.TicTacToe() for i in range(3)]for i in range(3)]
        self.occupancy=[[None for i in range(3)] for i in range(3)]
        self.nextSide=None
        self.nextBlock=None
        self.numFilledBlocks=0# Total number of blocks that can not be filled. 
        return super().__init__(**kwargs)

    def Take(self,**kwargs):
        """
        rtype: bool.  Return True if a side wins the UltimateTicTacToe. Return False if draw or not full.  
        """
        kwargs["rowBlock"]-=1
        kwargs["columnBlock"]-=1
        kwargs["rowSlot"]-=1
        kwargs["columnSlot"]-=1
        if not (kwargs["rowBlock"] in [0,1,2] and kwargs["columnBlock"] in [0,1,2]):
            raise ValueError("Invalid coordinate.")
        if (self.nextBlock!=None) and (self.blocks[kwargs["rowBlock"]][kwargs["columnBlock"]] is self.nextBlock)==False:
            raise ValueError("Rule violation. Invalid block")
        if self.blocks[kwargs["rowBlock"]][kwargs["columnBlock"]].occupancy==None:
            if (kwargs["side"]=="O" or kwargs["side"]=="X") and (self.nextSide==None or self.nextSide==kwargs["side"]):
                winStat=self.blocks[kwargs["rowBlock"]][kwargs["columnBlock"]].Take(kwargs["rowSlot"],kwargs["columnSlot"],kwargs["side"])
                self.nextSide="O" if kwargs["side"]=="X" else "X"
                self.nextBlock=None if self.blocks[kwargs["rowSlot"]][kwargs["columnSlot"]].occupancy!=None else self.blocks[kwargs["rowSlot"]][kwargs["columnSlot"]]
                if winStat==True:
                    #Search row, column and diagonal
                    self.numFilledBlocks+=1
                    if (self.blocks[kwargs["rowBlock"]][0].occupancy==self.blocks[kwargs["rowBlock"]][1].occupancy and self.blocks[kwargs["rowBlock"]][1].occupancy==self.blocks[kwargs["rowBlock"]][2].occupancy)\
                    or (self.blocks[0][kwargs["columnBlock"]].occupancy==self.blocks[1][kwargs["columnBlock"]].occupancy and self.blocks[1][kwargs["columnBlock"]].occupancy==self.blocks[2][kwargs["columnBlock"]].occupancy)\
                    or (kwargs["rowBlock"]==kwargs["columnBlock"] and self.blocks[0][0].occupancy==self.blocks[1][1].occupancy and self.blocks[1][1].occupancy==self.blocks[2][2].occupancy)\
                    or (kwargs["rowBlock"]+kwargs["columnBlock"]==2 and self.blocks[0][2].occupancy==self.blocks[1][1].occupancy and self.blocks[1][1].occupancy==self.blocks[2][0].occupancy):
                        self.occupancy=kwargs["side"]        
                        self.nextSide=None
                        self.nextBlock=None
                        return True
                    return False
                elif self.blocks[kwargs["rowBlock"]][kwargs["columnBlock"]].occupancy!=None:#draw
                    self.numFilledBlocks+=1
                    if self.numFilledBlocks==9:
                        self.occupancy="draw"
                    return False
            else:
                raise ValueError("Invalid input for side.")
        else:
            raise ValueError("Invalid block selection. Block occupied.")

    def __repr__(self):
        wholeUltimateTicTacToe=[]      
        for rowBlock in self.blocks:
            rowUltimateTicTacToeBlocks=[]
            for rowSlot in range(3):
                rowUltimateTicTacToe=[]
                for columnBlock in range(3):
                    rowUltimateTicTacToe.append(" | ".join(rowBlock[columnBlock].slots[rowSlot]))
                rowUltimateTicTacToeBlocks.append(" "+"||".join(rowUltimateTicTacToe)+" ")
            wholeUltimateTicTacToe.append("\n---------------------------------\n".join(rowUltimateTicTacToeBlocks))
        return "\n=================================\n".join(wholeUltimateTicTacToe)+"\n"

if __name__=="__main__":
    T=UltimateTicTacToe()
    print(T)
    T.Take(rowBlock=1,columnBlock=1,rowSlot=1,columnSlot=1,side="O")
    print(T)
    T.Take(rowBlock=1,columnBlock=1,rowSlot=3,columnSlot=1,side="X")
    print(T)
    T.Take(rowBlock=3,columnBlock=1,rowSlot=3,columnSlot=1,side="O")
    print(T)
    T.Take(rowBlock=3,columnBlock=1,rowSlot=1,columnSlot=1,side="X")
    print(T)
    T.Take(rowBlock=1,columnBlock=1,rowSlot=2,columnSlot=2,side="O")
    print(T)
    T.Take(rowBlock=2,columnBlock=2,rowSlot=1,columnSlot=1,side="X")
    print(T)
    T.Take(rowBlock=1,columnBlock=1,rowSlot=3,columnSlot=3,side="O")
    print(T)