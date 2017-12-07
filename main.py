from ultimate_tic_tac_toe import game_manager
from ultimate_tic_tac_toe.mcts import MCT
import configparser
import argparse
import os


def lets_play(config_parser):
    print("Welcome!")
    while True:
        try:
            rule = int(input("Select rule set! \n1: Standard \n2: Bizarre \n"))
        except Exception:  # might be fixed
            print("Invalid input for rule set. ")
        else:
            if rule in (1, 2):
                break
            else:
                continue
    model_name_mapping = (config_parser.get("model_config", "model_name_normal_rule"),
                          config_parser.get("model_config", "model_name_bizarre_rule"))
    model_path = os.path.normpath(os.path.join(config_parser.get("model_config", "model_dir"), model_name_mapping[rule - 1]))
    print("Loading model...")
    game = game_manager.GameManager(model_path)
    print("Done!")
    time_to_say_goodbye = False
    while not time_to_say_goodbye:
        while True:
            result = input(r"Do you want to play as initiator?[Y/N]").strip().lower()
            if result == "y":
                as_initiator = True
                break
            elif result == "n":
                as_initiator = False
                break
            else:
                print('The expected input is enter either Y or N, got "{}"'.format(result))
        while True:
            result = input(r"Which side do you want to play as?[O/X]").strip().lower()
            if result == "o":
                side = "O"
                break
            elif result == "x":
                side = "X"
                break
            else:
                print('The expected input is enter either O or X, got "{}"'.format(result))
        game.play_in_terminal(side, as_initiator, config_parser.getint("game_config", "computational_cost"),
                              config_parser.getboolean("game_config", "update_model_after_each_round"))
        while True:
            restart_request = input("Start another round?[Y/n]").strip().lower()
            if restart_request == "y":
                print("Let's try again! ")
                break
            elif restart_request == "n":
                time_to_say_goodbye = True
                print("Bye! ")
                break
            else:
                print("Unrecognized input, please try again. ")


def train(args, config_parser):
    model_name_mapping = {"normal": config_parser.get("model_config", "model_name_normal_rule"),
                          "bizarre": config_parser.get("model_config", "model_name_bizarre_rule")}
    model_path = os.path.normpath(os.path.join(config_parser.get("model_config", "model_dir"), model_name_mapping[args.rule]))
    tree = MCT.load_model(model_path)
    result = MCT.offline_learning(tree, args.epoch, args.verbose)
    print("#Exploitation: {}\n#Exploration: {}\n".format(*result))
    print(tree)
    MCT.save_model(tree, model_path)


if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config")
    arg_parser = argparse.ArgumentParser(description="Illustrations for parameters. ")
    arg_parser.add_argument("--train", action="store_true", default=False, help="Specify this to train the model. ")
    arg_parser.add_argument("--rule", action="store", type=str, help='Could be either "normal" or "bizarre". ')
    arg_parser.add_argument("--epoch", action="store", type=int, help="Specify the number of epochs in training. ")
    arg_parser.add_argument("--verbose", action="store", type=int, default=1,
                            help="Specify the verbosity level in training. ")
    args = arg_parser.parse_args()
    if not args.train:
        lets_play(config_parser)
    else:
        train(args, config_parser)
