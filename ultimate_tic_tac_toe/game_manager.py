from ultimate_tic_tac_toe.mcts import MCT
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe

class GameManager:
    def __init__(self, modelPath):
        self.__model = MCT.loadModel(modelPath)

    def playInTerminal(self,side="X", asInitiator=True):
        def outputStream():
            while True:
                yield tuple(int(ch) for ch in input("rowBlock,columnBlock,rowSlot,columnSlot\n").split(","))
        inputStream=MCT.onlineLearning(self.__model, outputStream(), side, asInitiator)
        terminal=False
        while terminal==False:
            try:
                action, board=next(inputStream)
                print("The opponent took action {}. ".format(action))
                print(board)
            except StopIteration:
                terminal=True


if __name__ == "__main__":
    modelPath=r"../model/test-rule1.pkl"
    GameManager(modelPath).playInTerminal("X", False)



