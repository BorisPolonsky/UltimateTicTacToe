import pickle
from UltimateTicTacToe import *
import time
class MCT:
    class NodeMCT:
        def __init__(self,**kwargs):
            """
            Initialization of a node.
            
            self.__state: the corresponding state of the UltimateTicTacToe board
            self.__prevNode: the parent node.(For back propogation)
            self.__val: a tuple of (#game in which the initialtor wins, #total games)
            self.__next: a list of children of the node.
            self.__prevAction: None for root. (rowBlock, rowColumn, rowSlot, columnSlot) for children nodes.
            """
            
            if ("state" in kwargs)==False:
                if "sovereignityUponDraw" in kwargs:
                    self.__state=pickle.dumps(UltimateTicTacToe(sovereignityUponDraw=kwargs["sovereignityUponDraw"]))
                else:
                    self.__state=pickle.dumps(UltimateTicTacToe())
            elif type(kwargs["state"])==UltimateTicTacToe:
                self.__state=pickle.dumps(kwargs["state"])
            else:
                raise TypeError("Invalid type of state.")
            if ("prevNode" in kwargs)==False:
                self.__prev=None#For back propogation
            elif type(kwargs["prevNode"])!=MCT.NodeMCT:
                raise TypeError("Invalid type of prevNode")
            if ("val" in kwargs)==False:
                self.__val=(0,0)
            self.__next=[]

        def addChild(self,action):
            if type(node)!=type(self):
                raise TypeError("node must be type of {}".format(type(self)))
            self.__next.append(node)
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
        for epoch in range(epochNum):
            node=tree


    def saveModel(tree,modelPath):
        if type(tree)!=MCT:
            raise TypeError("Invalid type of tree")
        try:
            with open(modelpath,"wb") as fileObj:
                pickle.dump(tree,fileObj)
        except IOError as e:
            print(e)

def loadModel(**kwargs):
    if "modelPath" in kwargs:
        try:
            with open(kwargs["modelPath"],"rb") as fileObj:
                return pickle.load(fileObj)
        except IOError as e:
            print(e)
    return None

if __name__=="__main__":
    T=MCT()
    T.root=MCT.NodeMCT()
    T.root.addChild(MCT.NodeMCT())
    with open (r".\Trees\test.pkl","wb") as fileObj:
        pickle.dump(T,fileObj)
    del T
    with open (r".\Trees\test.pkl","rb") as fileObj:
        T= pickle.load(fileObj)
    s=pickle.dumps(T)
    del T
    T=pickle.loads(s)
    del T
    T=loadModel(modelPath=r".\Trees\test.pkl")