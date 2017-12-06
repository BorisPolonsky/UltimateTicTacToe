from ultimate_tic_tac_toe.mcts import MCT


class GameManager:
    def __init__(self, model_path):
        self._model = MCT.load_model(model_path)
        self._model_path = model_path

    def play_in_terminal(self, side="X", as_initiator=True, num_of_eval=1000, learn=False):
        def output_stream():
            while True:
                try:
                    input_msg = input("Please enter your action with the following form:\nrowBlock,columnBlock,rowSlot,columnSlot\nNote that the commas can be excluded.\n")
                    if "," in input_msg:
                        action = tuple(int(ch) for ch in input_msg.split(","))
                    else:
                        action = tuple(int(ch) for ch in input_msg)
                except Exception as e:  # Need to fix it later
                    print(e)
                    print("Invalid input.")
                    continue
                yield action
        input_stream = MCT.onlineLearning(self._model, output_stream(), side, as_initiator, num_of_eval)
        while True:
            try:
                action, info = next(input_stream)
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
        if learn:
            print("Saving new model, please DON'T terminate the program...")
            MCT.save_model(self._model, self._model_path)
            print("Complete!")


if __name__ == "__main__":
    model_path = r"../model/bizarre.pkl"
    game = GameManager(model_path)
    game.play_in_terminal("X", False)



