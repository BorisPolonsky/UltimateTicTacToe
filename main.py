from ultimate_tic_tac_toe import game_manager
from ultimate_tic_tac_toe.mcts import MCT
if __name__ == "__main__":
    print("Welcome!")
    while True:
        try:
            rule=int(input("Select rule set! \n1: Standard \n2: Bizarre \n"))
        except Exception: #  might be fixed
            print("Invalid input for rule set. ")
        else:
            if rule in (1,2):
                break
            else:
                continue
    modelPath= r"./model/test-rule{}.pkl".format(rule)
    print("Loading model...")
    game=game_manager.GameManager(modelPath)
    print("Done!")
    while True:
        result=input(r"Do you want to play as initiator?[Y/N]")
        if result == "Y" or result == "y":
            asInitiator = True
            break
        elif result == "N" or result == "n":
            asInitiator = False
            break
        else:
            print('The expected input is enter either Y or N, got "{}"'.format(result))
    while True:
        result=input(r"Which side do you want to play as?[O/X]")
        if result=="O" or result=="o":
            side="O"
            break
        elif result=="X" or result=="x":
            side="X"
            break
        else:
            print('The expected input is enter either O or X, got "{}"'.format(result))
    game.playInTerminal(side, asInitiator)