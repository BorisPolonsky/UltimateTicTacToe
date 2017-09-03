from ultimate_tic_tac_toe.mcts import MCT
from ultimate_tic_tac_toe.game_board import UltimateTicTacToe

class GameManager:
    def __init__(self, modelPath):
        self.__model = MCT.loadModel(modelPath)

    def playInTerminal(self, side="X", asInitiator=True, numOfEval=1000):
        def outputStream():
            while True:
                try:
                    inputMsg=input("Please enter your action with the following form:\nrowBlock,columnBlock,rowSlot,columnSlot\nNote that the commas can be excluded.\n")
                    if "," in inputMsg:
                        action = tuple(int(ch) for ch in inputMsg.split(","))
                    else:
                        action = tuple(int(ch) for ch in inputMsg)
                except Exception as e:  # Need to fix it later
                    print(e)
                    print("Invalid input.")
                    continue
                yield action
        inputStream=MCT.onlineLearning(self.__model, outputStream(), side, asInitiator, numOfEval)
        while True:
            try:
                action, info = next(inputStream)
                if action is not None:
                    print("The opponent took action {}. \nScore: {}\nLog: {}".format(action, info["score"], info["log"]))
                    print(info["board"])
                else:  # In this case it's the user who ends the game.
                    print(info["board"])
                    break
            except StopIteration:
                break
        if info["board"].occupancy == "draw":
            print("Draw!")
        elif info["board"].occupancy == side:
            print("Congratulations! You win! ")
        else:
            print("You lose, please try again. ")

if __name__ == "__main__":
    modelPath=r"../model/test-rule1.pkl"
    game=GameManager(modelPath)
    game.playInTerminal("X", False)



