from ultimate_tic_tac_toe.mcts import MCT
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe

class GameManager:
    def __init__(self, modelPath):
        self.__model = MCT.loadModel(modelPath)

    def playInTerminal(self,side="X", asInitiator=True):
        def outputStream():
            while True:
                yield tuple(int(ch) for ch in input("rowBlock,columnBlock,rowSlot,columnSlot\n").split(","))
        MCT.onlineLearning(self.__model,outputStream(),side,asInitiator)

if __name__ == "__main__":
    modelPath=r"../model/test-rule1.pkl"
    GameManager(modelPath).playInTerminal("X",True)



