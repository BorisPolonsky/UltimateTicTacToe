import pickle
from UltimateTicTacToe import *
import time
import os
class MCT:
    class NodeMCT:
        def __init__(self, op= None):
            """
            Initialization of a node.
            self.__op: the operation from last state to reach this node.
            None for root, (rowBlock, rowColumn, rowSlot, columnSlot) for the rest.
            self.__val: a tuple of (#game in which the initiator wins+0.5*#draw, #total games)
            self.__next: a list of children of the node.
            """
            if op is None or (type(op)==tuple and len(op)==4):
                self.__op= op
            else:
                raise TypeError('Parameter "op" must be either None or a tuple of (rowBlock, rowColumn, rowSlot, columnSlot)')
            self.__val=(0,0)
            self.__next=[]

        def addChild(self,action):
            node=MCT.NodeMCT(action)
            self.__next.append(node)

        @property
        def children(self):
            return self.__next[:]

    def __init__(self):
        """
        Create an empty MCT.
        """
        self.__root=MCT.NodeMCT()

    def trainMCT(tree,epochNum=20):
        """
        epoch: Total number of game rounds simluated.
        """
        pass

    @classmethod
    def saveModel(cls, tree, modelPath):
        if type(tree)!=MCT:
            raise TypeError("Invalid type of tree")
        try:
            with open(os.path.normpath(modelPath),"wb") as fileObj:
                pickle.dump(tree,fileObj)
        except IOError as e:
            print(e)

def loadModel(**kwargs):
    if "modelPath" in kwargs:
        try:
            with open(os.path(kwargs["modelPath"]),"rb") as fileObj:
                return pickle.load(fileObj)
        except IOError as e:
            print(e)
    return None

if __name__=="__main__":
    pass