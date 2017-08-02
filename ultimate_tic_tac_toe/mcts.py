import pickle
import time
import os
import math
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe
import random
from enum import Enum,unique
class MCT:
    class NodeMCT:
        @unique
        class Result(Enum):
            initiatorWins=0
            initiatorLoses=1
            draw=2
        def __init__(self, op= None):
            """
            Initialization of a node.
            self.__op: the operation from last state to reach this node.
            None for root, (rowBlock, rowColumn, rowSlot, columnSlot) for the rest.
            self.__val: a tuple of (#game in which the initiator wins, #game in which the initiator loses, #total games)
            #draw=self.__val[2]-self.__val[0]-self.__val[1]
            self.__next: a list of children of the node.
            """
            if (type(op)==tuple and len(op)==4):
                for val in op:
                    if val not in range(1, 4):
                        raise ValueError("Each entry of the state must be an integer from 1-3")
                self.__op= op
            elif op is None:
                self.__op= op
            else:
                raise TypeError('Parameter "op" must be either None or a tuple of (rowBlock, rowColumn, rowSlot, columnSlot)')
            self.__val=(0,0,0)
            self.__next=[]

        def addChild(self, action):
            node = MCT.NodeMCT(action)
            self.__next.append(node)
            return node

        def update(self,result):
            if result==MCT.NodeMCT.Result.initiatorWins:
                self.__val= (self.__val[0]+1, self.__val[1], self.__val[2]+1)
            elif result==MCT.NodeMCT.Result.initiatorLoses:
                self.__val= (self.__val[0]+1, self.__val[1]+1, self.__val[2]+1)
            elif result==MCT.NodeMCT.Result.draw:
                self.__val= (self.__val[0], self.__val[1], self.__val[2]+1)  # ?
            else:
                raise ValueError("Invalid result. ")

        @property
        def children(self):
            return self.__next[:]

        @property
        def state(self):
            return self.__op

        @property
        def record(self):
            return self.__val

    @unique
    class Stage(Enum):
        selection=0
        simulation=1
        backPropogation=2

    def __init__(self, sovereignityUponDraw="none"):
        """
        Create an empty MCT.
        self.__root: root of the tree
        self.__size: total number of nodes (except root node) in the tree
        """
        self.__root=MCT.NodeMCT()
        self.__size = 0
        try:
            UltimateTicTacToe(sovereignityUponDraw=sovereignityUponDraw)
        except Exception as e:
            raise e
        else:
            self.__sovereignityUponDraw = sovereignityUponDraw

    @classmethod
    def trainMCT(cls, tree, epochNum=20):
        """
        epoch: Total number of game rounds simluated.
        """
        nExploration, nExploitation=0, 0
        for epoch in range(1,epochNum+1):
            print("Training epoch: {}".format(epoch))
            terminal=False
            currentNode=tree.__root
            initiator, opponent="X", "O"

            currentSide=initiator
            board=UltimateTicTacToe(sovereignityUponDraw=tree.__sovereignityUponDraw)
            stage=MCT.Stage.selection
            path=[currentNode]
            while not terminal:
                print(board)
                # Selection
                if stage == MCT.Stage.selection:
                    exploredActions=[node.state for node in currentNode.children]
                    validActions=board.validActions
                    actionsToBeExplored = list(set(validActions) - set(exploredActions))
                    if len(list(set(exploredActions)-set(validActions)))>0:
                        raise RuntimeError("Explored invalid actions!")
                    if len(actionsToBeExplored)>0:
                        action = random.choice(actionsToBeExplored)
                        stage=MCT.Stage.simulation
                        currentNode=tree.__addNode(currentNode, action)
                        path.append(currentNode)
                        nExploration+=1
                    else:
                        if currentSide == initiator:
                            scores = [node.record[0] / node.record[2] + math.sqrt(
                                2 * math.log(currentNode.record[2]) / node.record[2]) for node in currentNode.children]
                        else:
                            scores = [node.record[1] / node.record[2] + math.sqrt(
                                2 * math.log(currentNode.record[2]) / node.record[2]) for node in currentNode.children]
                        scoreNodePairs=zip(scores, currentNode.children)
                        # Select action with highest score
                        currentNode=max(scoreNodePairs, key=lambda x:x[0])[1]
                        path.append(currentNode)
                        action=currentNode.state
                        nExploitation+=1
                    print("Taking action {}".format(action))
                    terminal=board.take(*action, currentSide)

                elif stage == MCT.Stage.simulation:
                    action= board.randomAction
                    currentNode=tree.__addNode(currentNode, action)
                    print("Taking action {}".format(action))
                    terminal=board.take(*action, currentSide)
                    path.append(currentNode)
                    nExploration+=1

                currentSide="X" if currentSide=="O" else "O"
            # back-propagation
            if board.occupancy==initiator:
                result=MCT.NodeMCT.Result.initiatorWins
            elif board.occupancy==opponent:
                result=MCT.NodeMCT.Result.initiatorLoses
            elif board.occupancy=="draw":
                result=MCT.NodeMCT.Result.draw
            else:
                raise ValueError("Failed to get expected result.")
            for node in path:
                node.update(result)
        return nExploitation, nExploration

    def __addNode(self,parent,action):
        node=parent.addChild(action)
        self.__size+=1
        return node

    @property
    def size(self):
        return self.__size

    @classmethod
    def saveModel(cls, tree, modelPath):
        if type(tree)!=MCT:
            raise TypeError("Invalid type of tree")
        try:
            with open(os.path.normpath(modelPath),"wb") as fileObj:
                pickle.dump(tree,fileObj)
        except IOError as e:
            print(e)

    @classmethod
    def loadModel(cls,modelPath):
        try:
            with open(os.path.normpath(modelPath), "rb") as fileObj:
                return pickle.load(fileObj)
        except IOError as e:
            print(e)
            return None

    def __repr__(self):
        return "Knowledge base size: {}\nOverall: {}\n".format(self.size, self.__root.record)

if __name__ == "__main__":
    path1 = r"../model/test-rule1.pkl"
    path2 = r"../model/test-rule2.pkl"
    tree1 = MCT.loadModel(path1)
    if tree1 is None:
        tree1 = MCT(sovereignityUponDraw="none")
    print(tree1)
    result = MCT.trainMCT(tree1, 10000)
    print("#Explitation:{}\n#Exploration:{}\n".format(result[0], result[1]))
    print(tree1)
    MCT.saveModel(tree1, path1)
    tree2 = MCT.loadModel(path2)
    if tree2 is None:
        tree2 = MCT(sovereignityUponDraw="both")
    print(tree2)
    result = MCT.trainMCT(tree2, 10000)
    print("#Explitation:{}\n#Exploration:{}\n".format(result[0], result[1]))
    print(tree2)
    MCT.saveModel(tree2, path2)

