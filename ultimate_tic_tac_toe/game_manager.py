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
        while True:
            try:
                action, board=next(inputStream)
                if action is not None:
                    print("The opponent took action {}. ".format(action))
                    print(board)
                else:  # In this case it's the user who ends the game.
                    print(board)
                    break
            except StopIteration:
                break
        if board.occupancy == "draw":
            print("Draw!")
        elif board.occupancy == side:
            print("Congratulations! You win! ")
        else:
            print("You lose, please try again. ")

if __name__ == "__main__":
    modelPath=r"../model/test-rule1.pkl"
    game=GameManager(modelPath)
    game.playInTerminal("X", False)



