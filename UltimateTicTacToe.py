import copy
import random
class UltimateTicTacToe():
    class TicTacToe():
        def __init__(self, **kwargs):
            self.__slots=[[" "]*3 for i in range(3)]
            self.__occupancy=None#"O" , "X" or "draw"
            self.__numFilledSlots=0
            return super().__init__(**kwargs)

        def Take(self,row,column,side):
            """
            rtype: bool. Return True if:
            A side wins the TicTacToe or draw. 
            """
            if not (row in [0,1,2] and column in [0,1,2]):
                raise ValueError("Invalid coordinate.")
            if (self.__slots[row][column]==" "):
                if side=="O" or side=="X":
                    self.__numFilledSlots+=1
                    self.__slots[row][column]=side
                    #Search row, column and diagonal
                    if (self.__slots[row][0]==self.__slots[row][1] and self.__slots[row][1]==self.__slots[row][2])\
                    or (self.__slots[0][column]==self.__slots[1][column] and self.__slots[1][column]==self.__slots[2][column])\
                    or (row==column and self.__slots[0][0]==self.__slots[1][1] and self.__slots[1][1]==self.__slots[2][2])\
                    or (row+column==2 and self.__slots[0][2]==self.__slots[1][1] and self.__slots[1][1]==self.__slots[2][0]):
                        self.__occupancy=side
                        return True
                    if self.__numFilledSlots==9:
                        self.__occupancy="draw"
                        return True
                    return False
                else:
                    raise ValueError("Invalid input for side.")
            else:
                raise ValueError("The slot is already occupied!")

        def slot(self):
            return [row[:] for row in self.__slots]

        def occupancy(self):
            return self.__occupancy

        @property
        def validActions(self):
            """
            Return a list of actions : [(slotRow,slotColumn),...]
            Each index starts with 0.
            """
            if self.__occupancy!=None:
                return []
            #return [(row,column) for column in range(3) for row in range(3) if self.__slots[row][column] ==" "]
            actions=[]
            for row in range(3):
                for column in range(3):
                    if self.__slots[row][column] ==" ":
                        actions.append((row,column))
            return actions


    def __init__(self, **kwargs):
        self.__blocks=[[UltimateTicTacToe.TicTacToe() for i in range(3)]for i in range(3)]
        self.__occupancy=[[None for i in range(3)] for i in range(3)]
        self.__nextSide=None
        self.__nextBlock=None
        self.__numFilledBlocks=0# Total number of blocks that has been occupied.
        self.__occupancy=None
        if ("sovereignityUponDraw" in kwargs)==False:
            self.__sovereignityUponDraw="none"
        elif kwargs["sovereignityUponDraw"] in ("none","both"):
            self.__sovereignityUponDraw=kwargs["sovereignityUponDraw"]
        else:
            raise ValueError('Invalid rule set. "sovereignityUponDraw" should be either "both" or "none".')
        return super().__init__(**kwargs)

    def Take(self,**kwargs):
        """
        rtype: bool.  Return True if draw or a side wins the UltimateTicTacToe. Return False if the game is unfinished.  
        """
        kwargs["rowBlock"]-=1
        kwargs["columnBlock"]-=1
        kwargs["rowSlot"]-=1
        kwargs["columnSlot"]-=1
        if not (kwargs["rowBlock"] in [0,1,2] and kwargs["columnBlock"] in [0,1,2]):
            raise ValueError("Invalid coordinate.")
        if (self.__nextBlock!=None) and (self.__blocks[kwargs["rowBlock"]][kwargs["columnBlock"]] is self.__nextBlock)==False:
            raise ValueError("Rule violation. Invalid block")
        if self.__blocks[kwargs["rowBlock"]][kwargs["columnBlock"]].occupancy()==None:
            if (kwargs["side"]=="O" or kwargs["side"]=="X") and (self.__nextSide==None or self.__nextSide==kwargs["side"]):
                terminal=self.__blocks[kwargs["rowBlock"]][kwargs["columnBlock"]].Take(kwargs["rowSlot"],kwargs["columnSlot"],kwargs["side"])
                self.__nextSide="O" if kwargs["side"]=="X" else "X"
                self.__nextBlock=None if self.__blocks[kwargs["rowSlot"]][kwargs["columnSlot"]].occupancy()!=None else self.__blocks[kwargs["rowSlot"]][kwargs["columnSlot"]]
                if terminal==True:
                    self.__numFilledBlocks+=1
                    horizontal=(self.__blocks[kwargs["rowBlock"]][0].occupancy(),self.__blocks[kwargs["rowBlock"]][1].occupancy(),self.__blocks[kwargs["rowBlock"]][2].occupancy())
                    vertical=(self.__blocks[0][kwargs["columnBlock"]].occupancy(),self.__blocks[1][kwargs["columnBlock"]].occupancy(),self.__blocks[2][kwargs["columnBlock"]].occupancy())
                    diagonal1=(self.__blocks[0][0].occupancy(),self.__blocks[1][1].occupancy(),self.__blocks[2][2].occupancy())
                    diagonal2=(self.__blocks[2][0].occupancy(),self.__blocks[1][1].occupancy(),self.__blocks[0][2].occupancy())
                    if self.__sovereignityUponDraw=="both":
                        horizontal=map(lambda x:x if x!="draw" else kwargs["side"],horizontal)
                        vertical=map(lambda x:x if x!="draw" else kwargs["side"],vertical)
                        diagonal1=map(lambda x:x if x!="draw" else kwargs["side"],diagonal1)
                        diagonal2=map(lambda x:x if x!="draw" else kwargs["side"],diagonal2)
                    # Search row, column and diagonal
                    if (horizontal[0]==horizontal[1] and horizontal[1]==horizontal[2])\
                    or (vertical[0]==vertical[1] and vertical[1]==vertical[2])\
                    or (kwargs["rowBlock"]==kwargs["columnBlock"] and diagonal1[0]==diagonal1[1] and diagonal1[1]==diagonal1[2])\
                    or (kwargs["rowBlock"]+kwargs["columnBlock"]==2 and diagonal2[0]==diagonal2[1] and diagonal2[1]==diagonal2[2]):
                        self.__occupancy=kwargs["side"]        
                        self.__nextSide=None
                        self.__nextBlock=None
                        return True
                    elif self.__numFilledBlocks==9:
                        self.__occupancy=="draw"
                        self.__nextSide=None
                        self.__nextBlock=None
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
    def validActions(self):
        """
        Return a list of actions : [(blockRow,blockColumn,slotRow,slotColumn),...]
        Each index starts with 1.
        """
        actions=[]
        for row in range(3):
            for column in range(3):
                for coordinate in self.__blocks[row][column].validActions:
                    actions.append((row+1, column+1, coordinate[0]+1, coordinate[1]+1))
        return actions

    @property
    def randomAction(self):
        actions=self.validActions
        if len(actions)>0:
            return random.choice(actions)
        else:
            return None

    def occupancy(self):
        return self.__occupancy

    def __repr__(self):
        wholeUltimateTicTacToe=[]      
        for rowBlock in self.__blocks:
            rowUltimateTicTacToeBlocks=[]
            for rowSlot in range(3):
                rowUltimateTicTacToe=[]
                for columnBlock in range(3):
                    rowUltimateTicTacToe.append(" | ".join(rowBlock[columnBlock].slot()[rowSlot]))
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
    T.Take(rowBlock=3,columnBlock=3,rowSlot=1,columnSlot=1,side="X")
    print(T)
    T.Take(rowBlock=2,columnBlock=2,rowSlot=3,columnSlot=3,side="O")
    print(T)
    print(T.validActions)
    print(T.randomAction)
    print(T.occupancy)