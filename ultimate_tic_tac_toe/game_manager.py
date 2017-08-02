from ultimate_tic_tac_toe.mcts import MCT
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe

class GameManager:
    def __init__(self, modelPath):
        self.__model = MCT.loadModel(modelPath)

    def playInTerminal(self,side="X", asInitiator=True):
        def outputStream():
            while True:
                try:
                    action=tuple(int(ch) for ch in input("Please enter your action with the follwing form:\nrowBlock,columnBlock,rowSlot,columnSlot\n").split(","))
                except Exception as e:  # Need to fix it later
                    print(e)
                    print("Invalid input.")
                    continue
                yield action
        inputStream=MCT.onlineLearning(self.__model, outputStream(), side, asInitiator, 50)
        terminal=False
        while terminal==False:
            try:
                action, board=next(inputStream)
                print("The opponent took action {}. ".format(action))
                print(board)
            except StopIteration:
                terminal=True
        if board.occupancy == "draw":
            print("Draw!")
        elif board.occupancy==side:
            print("Congratulations! You win! ")
        else:
            print("You lose, please try again. ")

if __name__ == "__main__":
    modelPath=r"../model/test-rule1.pkl"
    game=GameManager(modelPath)
    game.playInTerminal("X", False)



