import pickle
import time
import os
import math
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe
import random
from enum import Enum, unique
import copy
class MCT:
    class NodeMCT:
        @unique
        class Result(Enum):
            initiatorWins = 0
            initiatorLoses = 1
            draw = 2
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

        # Need to be checked
        def addChild(self, action):
            node = MCT.NodeMCT(action)
            self.__next.append(node)
            return node

        def update(self, result):
            if result==MCT.NodeMCT.Result.initiatorWins:
                self.__val= (self.__val[0]+1, self.__val[1], self.__val[2]+1)
            elif result==MCT.NodeMCT.Result.initiatorLoses:
                self.__val= (self.__val[0]+1, self.__val[1]+1, self.__val[2]+1)
            elif result==MCT.NodeMCT.Result.draw:
                self.__val= (self.__val[0], self.__val[1], self.__val[2]+1)  # ?
            else:
                raise ValueError("Invalid result. ")

        def getBestChild(self,isInitiator, c=math.sqrt(2)):
            if c<0:
                raise ValueError('Parameter c must be greater or equal to 0. ')
            children=self.children
            if isInitiator:
                scoreNodePairs = [
                    (node.record[0] / node.record[2] + c * math.sqrt(math.log(self.record[2]) / node.record[2]), node)
                    for node in children]
            else:
                scoreNodePairs = [
                    (node.record[1] / node.record[2] + c * math.sqrt(math.log(self.record[2]) / node.record[2]), node)
                    for node in children]
            return max(scoreNodePairs,key=lambda x:x[0], default=(None, None))[1]

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
    def offlineLearning(cls, tree, epochNum=1000):
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
                        currentNode=currentNode.getBestChild(currentSide==initiator)
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

    @classmethod
    def onlineLearning(cls, model, inputStream, side="X", asInitiator=True, numEvalForEachStep=20):
        """
        :param side: "X" or "O"
        :param asInitiator: bool. True if the input serves as the initiator the game.
        :return:
        Online learning with MCTS.
        """
        if side in ["X","O"]:
            opponent="X" if side=="O" else "O"
        else:
            raise ValueError('Invalid input for parameter "side", expected "X" or "O", got {}. '.format(side))
        if numEvalForEachStep<0:
            raise ValueError('Parameter "numEvalForEachStep" must be greater than 0. ')
        initiatorSide, defenderSide = (side, opponent) if asInitiator else (opponent, side)
        currentSide=initiatorSide
        terminal=False
        board=UltimateTicTacToe(sovereignityUponDraw=model.ruleSet["sovereignityUponDraw"])
        currentNode=model.__root
        nodePath=[model.__root]
        while not terminal:
            if currentSide == side:  # The input's turn
                while True:
                    try:
                        action = next(inputStream)
                    except StopIteration:
                        raise RuntimeError("Can't receive action from player. ")
                    if action not in board.validActions:
                        print("Invalid action. Please try again.")
                        continue
                    else:
                        terminal = board.take(*action, currentSide)
                        break
                for node in currentNode.children:
                    if node.state == action:
                        currentNode = node
                    else:
                        currentNode = currentNode.addChild(action)
                    nodePath.append(currentNode)
            else:  # The model's turn
                # To be fixed
                for testEpoch in range(100):
                    testTerminal=False
                    testCurrentNode = currentNode
                    testBoard=copy.deepcopy(board)
                    testCurrentSide=currentSide
                    testNodePath=[]
                    # Selection
                    while not testTerminal:
                        validActions=testBoard.validActions
                        exploredActions= [node.state for node in testCurrentNode.children]
                        actionsToBeExplored=list(set(validActions)-set(exploredActions))
                        if len(actionsToBeExplored)>0:
                            action=random.choice(actionsToBeExplored)
                            testCurrentNode= testCurrentNode.addChild(action)
                            testNodePath.append(testCurrentNode)
                            testTerminal= testBoard.take(*action, testCurrentSide)
                            testCurrentSide= testBoard.nextSide
                            break
                        else:
                            testCurrentNode=testCurrentNode.getBestChild(testCurrentSide==initiatorSide)
                            testNodePath.append(testCurrentNode)
                            action=testCurrentNode.state
                            testTerminal=testBoard.take(*action, testCurrentSide)
                            testCurrentSide= testBoard.nextSide

                    # Simulation
                    while not testTerminal:
                        action = testBoard.randomAction
                        testCurrentNode = testCurrentNode.addChild(action)
                        testNodePath.append(testCurrentNode)
                        testTerminal = testBoard.take(*action,testCurrentSide)
                        testCurrentSide = testBoard.nextSide
                    # Back-propagation
                    occupancy = testBoard.occupancy
                    if occupancy == "draw":
                        result=MCT.NodeMCT.Result.draw
                    elif occupancy == initiatorSide:
                        result=MCT.NodeMCT.Result.initiatorWins
                    elif occupancy==defenderSide:
                        result=MCT.NodeMCT.Result.initiatorLoses
                    else:
                        raise ValueError("Invalid occupancy when the game terminates.")
                    for node in nodePath+testNodePath:
                        node.update(result)
                currentNode=currentNode.getBestChild(currentSide==initiatorSide)
                action = currentNode.state
                nodePath.append(currentNode)
                board.take(*action, opponent)
                yield action, copy.deepcopy(board)
            currentSide="X" if currentSide=="O" else "O"

        # Final back-propagation
        occupancy = board.occupancy
        if occupancy == "draw":
            result = MCT.NodeMCT.Result.draw
        elif occupancy == initiatorSide:
            result = MCT.NodeMCT.Result.initiatorWins
        elif occupancy == defenderSide:
            result = MCT.NodeMCT.Result.initiatorLoses
        else:
            raise ValueError("Invalid occupancy when the game terminates.")
        for node in nodePath:
            node.update(result)
        #  Yield last result if it's the input who ended the game
        if currentSide == opponent:
            yield None, board

    def __addNode(self,parent,action):
        node=parent.addChild(action)
        self.__size+=1
        return node

    @property
    def size(self):
        return self.__size

    @property
    def ruleSet(self):
        return {"sovereignityUponDraw":self.__sovereignityUponDraw}

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
    def loadModel(cls, modelPath):
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
    result = MCT.offlineLearning(tree1, 10000)
    print("#Explitation:{}\n#Exploration:{}\n".format(result[0], result[1]))
    print(tree1)
    MCT.saveModel(tree1, path1)
    tree2 = MCT.loadModel(path2)
    if tree2 is None:
        tree2 = MCT(sovereignityUponDraw="both")
    print(tree2)
    result = MCT.offlineLearning(tree2, 10000)
    print("#Explitation:{}\n#Exploration:{}\n".format(result[0], result[1]))
    print(tree2)
    MCT.saveModel(tree2, path2)

