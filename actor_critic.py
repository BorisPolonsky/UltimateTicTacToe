from ultimate_tic_tac_toe.game_board import UltimateTicTacToe, BoardState
import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import glob
import re
import os


class ExponentialMovingAverage:
    def __init__(self, alpha: float):
        assert 0 < alpha <= 1
        self._ema = None
        self._alpha = alpha

    def update(self, value):
        if self._ema is None:
            self._ema = value
        else:
            alpha = self._alpha
            self._ema = (1 - alpha) * self._ema + alpha * value

    def result(self):
        return self._ema


class SimpleMovingAverage:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._buff = []

    def update(self, value):
        self._buff.append(value)
        if len(self._buff) > self._window_size:
            self._buff = self._buff[-self._window_size:]

    def result(self):
        return sum(self._buff) / len(self._buff)


class PolicyHead(torch.nn.Module):
    def __init__(self, feature_map_dim):
        super().__init__()
        n_filter_conv1 = 2
        self.conv1 = torch.nn.Conv2d(in_channels=feature_map_dim,
                                     out_channels=n_filter_conv1,
                                     kernel_size=1, stride=1, padding=0)
        self.activation1 = torch.nn.ReLU(inplace=False)
        self.fc = torch.nn.Conv2d(in_channels=n_filter_conv1,
                                  out_channels=81,
                                  kernel_size=9, stride=1, padding=0)
        # self.policy_activation = torch.nn.ReLU(inplace=False)
        self.fc_activation = torch.nn.Identity()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.activation1(out)
        out = self.fc(out)  # [B, N_CLS, 1, 1]
        out = self.fc_activation(out)
        out = out.flatten(start_dim=1)  # [B, N_CLS]
        return out


class ValueHead(torch.nn.Module):
    def __init__(self, feature_map_dim):
        super().__init__()
        n_filter_conv1 = 1
        self.conv1 = torch.nn.Conv2d(in_channels=feature_map_dim,
                                     out_channels=n_filter_conv1,
                                     kernel_size=1, stride=1, padding=0)
        self.conv1_activation = torch.nn.ReLU(inplace=False)
        hidden_dim = 256
        self.fc1 = torch.nn.Conv2d(in_channels=n_filter_conv1,
                                   out_channels=hidden_dim,
                                   kernel_size=9, stride=1, padding=0)
        self.fc1_activation = torch.nn.ReLU(inplace=False)
        self.fc2 = torch.nn.Conv2d(in_channels=hidden_dim,
                                   out_channels=1,
                                   kernel_size=1, stride=1, padding=0)
        # self.fc2_activation = torch.nn.Identity()
        self.fc2_activation = torch.nn.Tanh()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv1_activation(out)
        out = self.fc1(out)
        out = self.fc1_activation(out)
        out = self.fc2(out)
        out = self.fc2_activation(out)
        out = out.flatten(start_dim=0)  # [B]
        return out


class ActorCritic(torch.nn.Module):
    """
    Return policy for the current player and value for player 1 given board state    """
    def __init__(self, input_step=2):
        super().__init__()
        n_channel_board_layout_repr = 64
        n_channel_next_player_repr = 64
        self.input_step = input_step
        n_state_channel = n_channel_board_layout_repr * input_step + n_channel_next_player_repr
        self.board_layout_embedding = torch.nn.Embedding(3, embedding_dim=n_channel_board_layout_repr)
        self.next_player_embedding = torch.nn.Embedding(2, embedding_dim=n_channel_next_player_repr)
        n_filter_conv1 = 256
        self.conv1 = torch.nn.Conv2d(in_channels=n_state_channel,
                                     out_channels=n_filter_conv1,
                                     kernel_size=3, stride=3, padding=0)
        self.conv1_activation = torch.nn.GELU()
        n_filter_conv2 = 512
        self.conv2 = torch.nn.Conv2d(in_channels=n_filter_conv1,
                                     out_channels=n_filter_conv2,
                                     kernel_size=1, stride=1, padding=0)
        self.conv2_activation = torch.nn.GELU()
        n_filter_fpn_conv1 = n_filter_conv2
        self.upscale = torch.nn.Upsample(scale_factor=3, mode='nearest')
        self.fpn_conv1 = torch.nn.Conv2d(in_channels=n_state_channel + n_filter_conv2,
                                         out_channels=n_filter_fpn_conv1,
                                         kernel_size=1)
        self.fpn_conv1_activation = torch.nn.GELU()
        self.policy_head = PolicyHead(n_filter_conv2)
        self.value_head = ValueHead(n_filter_conv2)

    def forward(self, board_layout: torch.Tensor, next_player: torch.Tensor):
        """

        :param board_layout: torch.Tensor [batch_size, height, width, n_step]
        :param next_player: torch.Tensor [batch_size]
        :return:
        """
        board_layout_feature = self.board_layout_embedding(board_layout)  # [B, H, W, L, C']
        board_layout_feature = torch.flatten(board_layout_feature, 3)  # [B, H, W, C]
        next_player_feature = self.next_player_embedding(next_player.view(-1, 1, 1).repeat(1, 9, 9))
        board_state = torch.cat([board_layout_feature, next_player_feature], dim=-1)
        board_state = board_state.permute([0, 3, 1, 2])  # [B, C, H, W]
        out = self.conv1(board_state)
        out = self.conv1_activation(out)
        out = self.conv2(out)
        out = self.conv2_activation(out)
        # FPN
        out = self.upscale(out)
        out = torch.cat([board_state, out], dim=1)
        out = self.fpn_conv1(out)
        feature_map = out = self.fpn_conv1_activation(out)
        policy_logits = self.policy_head(feature_map)
        value = self.value_head(feature_map)  # [B]
        return policy_logits, value


def action2id(row_block: int, column_block: int, row_slot: int, column_slot: int):
    row = row_block * 3 + row_slot
    column = column_block * 3 + column_slot
    idx = row * 9 + column
    return idx


def id2action(idx: int) -> Tuple[int, int, int, int]:
    row, column = divmod(idx, 9)
    row_block, row_slot = divmod(row, 3)
    column_block, column_slot = divmod(column, 3)
    return row_block, column_block, row_slot, column_slot


def get_checkpoints(checkpoint_dir: str):
    checkpoints = []
    for path in glob.iglob(os.path.join(checkpoint_dir, "checkpoint-*")):
        match = re.search("checkpoint-([0-9]*).(pt)$", path)
        if match:
            n_iter = int(match.group(1))
            checkpoints.append((n_iter, path))
    return checkpoints


def get_last_checkpoint(checkpoint_dir: str):
    checkpoints = get_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise ValueError("No state_dict found in ".format(checkpoint_dir))
    n_iter, last_checkpoint = max(checkpoints, key=lambda x: x[0])
    return n_iter, last_checkpoint


def get_model_builder(input_step):
    def model_builder():
        return ActorCritic(input_step=input_step)
    return model_builder


def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_step_size = 2
    model_builder = get_model_builder(input_step_size)
    if args.do_train:
        use_gpu = True
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        train_fn(model_builder, output_dir, device, input_step_size)

    if args.do_eval:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        eval_fn(model_builder, output_dir, device, input_step_size, n_episode=500)
    if args.do_visualize:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        visualize(model_builder, output_dir, device, input_step_size)
    if args.do_interactive_eval:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        interactive_test(model_builder, output_dir, device, input_step_size)


def train_fn(model_builder, output_dir, device, n_input_step: int):
    nn = model_builder()
    nn_current_best = model_builder()
    try:
        ckpt_n_episode, path = get_last_checkpoint(output_dir)
        checkpoint = torch.load(path, map_location=device)
    except ValueError:
        print("No state dict found, training from scratch.")
        nn_current_best.load_state_dict(nn.state_dict())
        ckpt_n_episode = 0
    else:
        print("Loading state dict from {}".format(path))
        nn.load_state_dict(checkpoint["edge"])
        nn_current_best.load_state_dict(checkpoint["current_best"])
    nn.to(device)
    nn_current_best.to(device)
    num_batch = 1000000
    gamma = 0.98
    # optimizer = torch.optim.SGD(nn.parameters(), lr=2e-4)
    optimizer = torch.optim.AdamW(nn.parameters(), lr=2e-5)
    # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
    alpha_ema = 0.3
    alpha_value_loss = 10
    beta_policy_regularization = 0
    sma_window_size = 100
    log_every = 100
    eval_every = 5000
    save_checkpoints_per_n_episode = 10000  # How often to save state dict
    save_checkpoints_per_n_seconds = 1 * 3600
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o755)
    with SummaryWriter(os.path.join(output_dir, "log")) as writer:
        tic = time.time()
        reward_ema = ExponentialMovingAverage(alpha_ema)
        num_episode = prev_save_idx = ckpt_n_episode
        n_episode_current_generation = prev_log_idx = prev_eval_idx = 0
        generation = 0
        batch_size = 128
        for batch_i in range(num_batch):
            log_prob_history_1, log_prob_history_2 = [], []
            value_history = []
            value_history_1, value_history_2 = [], []
            policy_logits_history = []
            g = []
            g1, g2 = [], []
            for i in range(batch_size):
                num_action = 0
                board_layout_history = []
                is_terminal = False
                is_initiators_turn = True
                board = UltimateTicTacToe.create_initial_board()
                state_buff = [board.get_state() for _ in range(n_input_step - 1)]
                while not is_terminal:
                    if is_initiators_turn:
                        log_prob_history = log_prob_history_1
                        current_player_value_history = value_history_1
                    else:
                        log_prob_history = log_prob_history_2
                        current_player_value_history = value_history_2
                    board_layout = board.get_state()
                    board_layout_history.append(board_layout)
                    state_buff.append(board_layout)
                    state_buff = state_buff[-n_input_step:]
                    num_action += 1
                    batch_board_layout = torch.from_numpy(np.stack(state_buff, axis=-1)).to(dtype=torch.int64, device=device).unsqueeze(0)
                    batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
                    valid_actions = sorted(map(lambda x: action2id(*x), board.valid_actions))
                    valid_actions = torch.tensor(valid_actions, device=device)
                    policy_logits, values = nn(batch_board_layout, batch_next_player)
                    valid_policy_logits = policy_logits[:, valid_actions]
                    # create a categorical distribution over the list of probabilities of actions
                    valid_probs = torch.softmax(valid_policy_logits, dim=-1)
                    # print(batch, policy_logits, valid_probs)
                    m1 = torch.distributions.Categorical(valid_probs)

                    # and sample an action using the distribution
                    action = m1.sample()
                    # log_prob = m1.log_prob(action) # Masked log_prob
                    action = valid_actions[action]
                    m2 = torch.distributions.Categorical(torch.softmax(policy_logits, dim=-1))
                    log_prob = m2.log_prob(action)  # Unmasked log_prob

                    is_terminal = board.take(*id2action(action.squeeze().item()), side=1 if is_initiators_turn else 2)
                    policy_logits_history.append(policy_logits)
                    log_prob_history.append(log_prob)
                    value_history.append(values)
                    current_player_value_history.append(values)

                    is_initiators_turn = not is_initiators_turn

                # Calculate reward
                reward = 0
                if board.occupancy == BoardState.OCCUPIED_BY_PLAYER1:
                    reward = 1
                elif board.occupancy == BoardState.OCCUPIED_BY_PLAYER2:
                    reward = -1
                reward_ema.update(reward)
                reward_to_go = []
                r = reward
                for _ in range(num_action):
                    reward_to_go.insert(0, r)
                    r = r * gamma
                reward_to_go = torch.tensor(reward_to_go, device=device)
                g.append(reward_to_go)
                g1.append(reward_to_go[torch.arange(0, reward_to_go.size(0), 2)])
                g2.append(-reward_to_go[torch.arange(1, reward_to_go.size(0), 2)])
                del reward_to_go

            num_episode += batch_size
            n_episode_current_generation += batch_size

            log_prob_history_1 = torch.cat(log_prob_history_1)
            log_prob_history_2 = torch.cat(log_prob_history_2)
            g = torch.cat(g)
            g1 = torch.cat(g1)
            g2 = torch.cat(g2)
            value_history = torch.cat(value_history)
            value_history_1 = torch.cat(value_history_1)
            value_history_2 = -torch.cat(value_history_2)  # value_for_p2 = -value_for_p1
            policy_logits_history = torch.cat(policy_logits_history, dim=0)
            policy_entropy = torch.distributions.Categorical(torch.softmax(policy_logits_history, dim=-1)).entropy()
            # state_history = torch.from_numpy(np.stack(state_history))
            # state_history_1 = state_history[torch.arange(0, state_history.size(0), 2)]
            # state_history_2 = state_history[torch.arange(1, state_history.size(0), 2)]
            p1_policy_loss = -torch.sum(torch.mul(log_prob_history_1, (g1 - value_history_1.detach()))) / batch_size
            p2_policy_loss = -torch.sum(torch.mul(log_prob_history_2, (g2 - value_history_2.detach()))) / batch_size
            p_loss = p1_policy_loss + p2_policy_loss
            v_loss = F.smooth_l1_loss(value_history, g, reduction="mean", beta=0.05)
            entropy_regularization_term = -torch.sum(policy_entropy) / batch_size
            # Add illegal action loss?
            loss = p_loss + alpha_value_loss * v_loss + beta_policy_regularization * entropy_regularization_term
            # reset gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            toc = time.time()

            del board_layout_history, value_history_1, value_history_2, log_prob_history_1, log_prob_history_2
            if n_episode_current_generation - prev_log_idx > log_every:
                prev_log_idx = n_episode_current_generation
                writer.add_scalar("player1_policy_loss", p1_policy_loss.item(), num_episode)
                writer.add_scalar("player2_policy_loss", p2_policy_loss.item(), num_episode)
                writer.add_scalar("policy_loss", p_loss.item(), num_episode)
                writer.add_scalar("value_loss", v_loss.item(), num_episode)
                writer.add_scalar("total_loss", loss.item(), num_episode)
                writer.add_histogram("policy_entropy", policy_entropy, num_episode)  # max: 4.3944
                writer.add_scalar("value_explained_variance", 1 - (value_history.detach() - g).var() / g.var(), num_episode)
                print("loss", loss.item(), "policy_loss", p_loss.item(), "p1_policy_loss", p1_policy_loss.item(),  "p2_policy_loss", p2_policy_loss.item(), "v_loss", v_loss.item())
                print("Reward for Player1 (EMA):", reward_ema.result())
            if (n_episode_current_generation - prev_eval_idx) > eval_every:
                prev_eval_idx = n_episode_current_generation
                n_eval_per_role = 400
                nn_as_p1_result = eval_network(nn, n_input_step, nn_current_best, n_input_step, device, n_eval_per_role)
                nn_as_p2_result = eval_network(nn_current_best, n_input_step, nn, n_input_step, device, n_eval_per_role)
                nn_win_rate = (nn_as_p1_result["n_p1_wins"] + nn_as_p2_result["n_p2_wins"]) / (2 * n_eval_per_role)
                nn_as_p1_win_rate = nn_as_p1_result["n_p1_wins"] / n_eval_per_role
                nn_as_p2_win_rate = nn_as_p2_result["n_p2_wins"] / n_eval_per_role
                current_best_nn_win_rate = (nn_as_p1_result["n_p2_wins"] + nn_as_p2_result["n_p1_wins"]) / (2 * n_eval_per_role)
                smoothed_nn_win_rate = (nn_as_p1_result["n_p1_wins"] + nn_as_p2_result["n_p2_wins"] + 0.5 * (nn_as_p1_result["n_draw"] + nn_as_p2_result["n_draw"])) / (2 * n_eval_per_role)
                print("win_rate_as_p1", nn_as_p1_win_rate, "win_rate_as_p2", nn_as_p2_win_rate, "smoothed_nn_win_rate", smoothed_nn_win_rate)
                accept_new_network = (current_best_nn_win_rate == 0 and nn_win_rate > 0) or smoothed_nn_win_rate > 0.55 # nn_win_rate / current_best_nn_win_rate > 1.2
                if accept_new_network:
                    nn_current_best.load_state_dict(nn.state_dict())
                    generation += 1
                    writer.add_scalar("win_rate_as_p1", nn_as_p1_win_rate, num_episode)
                    writer.add_scalar("win_rate_as_p2", nn_as_p2_win_rate, num_episode)
                    writer.add_scalar("nn_generation", generation, num_episode)
                    writer.add_histogram("n_episode_trained_for_new_generation", n_episode_current_generation, num_episode)
                    print("Generation: {} win rate as p1: {}, win rate as p2: {}, overall win rate: {}. Baseline model win_rate: {}".format(generation, nn_as_p1_win_rate, nn_as_p2_win_rate, nn_win_rate, current_best_nn_win_rate))
                    reward_ema = ExponentialMovingAverage(alpha_ema)
                    n_episode_current_generation = prev_log_idx = prev_eval_idx = 0

            if (toc - tic) > save_checkpoints_per_n_seconds or (num_episode - prev_save_idx > save_checkpoints_per_n_episode):
                tic = time.time()
                checkpoint = {"current_best": nn_current_best.state_dict(), "edge": nn.state_dict()}
                torch.save(checkpoint, os.path.join(output_dir, "checkpoint-{}.pt".format(num_episode)))
                prev_save_idx = num_episode
                del checkpoint


def eval_fn(model_builder, output_dir, device, input_step_size, n_episode):
    nn = model_builder()
    ckpt_n_episode, path = get_last_checkpoint(output_dir)
    checkpoint = torch.load(path, map_location=device)
    nn.load_state_dict(checkpoint["current_best"])
    nn.to(device)
    nn.eval()
    nn1 = nn2 = nn
    nn1_input_step_size = nn2_input_step_size = input_step_size
    result = eval_network(nn1, nn1_input_step_size, nn2, nn2_input_step_size, device, n_episode=n_episode)
    print(result)


def visualize(model_builder, output_dir, device, input_step_size: int):
    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    board_layout_cmap = "RdYlBu"
    fig = plt.figure(figsize=(15, 10))
    ax_next_player = fig.add_subplot(521)
    ax_board_layout = fig.add_subplot(523)
    ax_logits = fig.add_subplot(522)
    ax_probs = fig.add_subplot(524)
    ax_action = fig.add_subplot(525)
    ax_probs_valid = fig.add_subplot(526)
    ax_entropy = fig.add_subplot(514)
    ax_values = fig.add_subplot(515)
    ax_next_player.set_title("Next Player")
    ax_board_layout.set_title("Board Layout")
    ax_logits.set_title("Logits from Policy Network")
    ax_probs.set_title(r"$\pi(\mathbf{a}|\mathbf{s})$ (w/o Valid Action Mask) from Policy Network")
    ax_probs_valid.set_title(r"$\pi(\mathbf{a}|\mathbf{s})$ (with Valid Action Mask) from Policy Network")
    ax_action.set_title("Action Taken")
    nn = model_builder()
    ckpt_n_episode, path = get_last_checkpoint(output_dir)
    checkpoint = torch.load(path, map_location=device)
    nn.load_state_dict(checkpoint["edge"])
    nn.to(device)
    nn.eval()
    nn1 = nn2 = nn
    is_terminal = False
    is_initiators_turn = True
    board = UltimateTicTacToe.create_initial_board()
    action_history = []
    board_layout_history = [board.get_state() for _ in range(input_step_size - 1)]
    valid_policy_entropy_history, valid_policy_entropy_history_player1, valid_policy_entropy_history_player2 = [], [], []
    policy_entropy_history = []
    value_history = []
    while not is_terminal:
        if is_initiators_turn:
            nn = nn1
        else:
            nn = nn2
        board_layout = board.get_state()
        board_layout_history.append(board_layout)
        batch_board_layout = torch.from_numpy(np.stack(board_layout_history[-input_step_size:], axis=-1)).to(dtype=torch.int64, device=device).unsqueeze(0)
        valid_actions = board.valid_actions
        valid_actions = torch.tensor(sorted(map(lambda x: action2id(*x), valid_actions)), device=device)
        valid_action_mask = torch.tensor([idx in set(map(lambda x: action2id(*x), board.valid_actions)) for idx in range(81)], device=device)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        print("Board :\n{}\nValue: {}".format(board.as_str(), values.squeeze().item()))
        valid_policy_logits = policy_logits[:, valid_actions]
        # p_action_val = policy_logits[0].tolist()
        # create a categorical distribution over the list of probabilities of actions
        valid_probs = torch.softmax(valid_policy_logits, dim=-1)
        print(batch_board_layout.squeeze(0), policy_logits.view(-1, 9, 9), "valid_action_logits", valid_policy_logits, "valid_action_probs", valid_probs)
        ax_next_player.imshow(np.full([9, 9], 1 if is_initiators_turn else -1), vmin=-1, vmax=1, cmap=board_layout_cmap)
        ax_next_player.xaxis.set_tick_params(length=0)
        ax_next_player.yaxis.set_tick_params(length=0)
        ax_board_layout.imshow(np.where(board_layout != 2, board_layout, np.full_like(board_layout, -1)), vmin=-1, vmax=1, cmap=board_layout_cmap)
        ax_logits.imshow(policy_logits.view(9, 9).numpy())
        ax_probs.imshow(torch.softmax(policy_logits, dim=-1).view(9, 9).numpy(), vmin=0, vmax=1)
        ax_probs_valid.imshow(torch.softmax(torch.where(valid_action_mask, policy_logits, torch.full_like(policy_logits, -50000)), dim=-1).view(9, 9), vmin=0, vmax=1)
        m1 = torch.distributions.Categorical(valid_probs)

        # and sample an action using the distribution
        action = m1.sample()

        ax_entropy.cla()
        ax_entropy.set_title(r"Entropy of $\pi(\mathbf{a}|\mathbf{s})$")
        entropy_raw = torch.distributions.Categorical(torch.softmax(policy_logits, dim=-1)).entropy().item()
        entropy_valid = m1.entropy()
        policy_entropy_history.append(entropy_raw)
        valid_policy_entropy_history.append(entropy_valid)
        if is_initiators_turn:
            valid_policy_entropy_history_player1.append(entropy_valid)
        else:
            valid_policy_entropy_history_player2.append(entropy_valid)
        ax_entropy.scatter(np.arange(0, 2 * len(valid_policy_entropy_history_player1), 2), valid_policy_entropy_history_player1, color="c", label="Player 1")
        ax_entropy.scatter(np.arange(1, 1 + 2 * len(valid_policy_entropy_history_player2), 2), valid_policy_entropy_history_player2, color="r", label="Player 2")
        ax_entropy.plot(policy_entropy_history, label="with valid action mask")
        ax_entropy.plot(valid_policy_entropy_history, label="w/o valid action mask")

        ax_entropy.legend()
        value_history.append(values.squeeze().item())
        ax_values.cla()
        ax_values.set_title(r"$v(\mathbf{s})$ for Player 1")
        ax_values.plot(value_history)

        action = valid_actions[action]
        action_id = action.squeeze().item()
        action_history.append(action_id)
        ax_action.imshow(np.where(np.arange(81).reshape(9, 9) == action_id, 1 if is_initiators_turn else -1, 0), vmin=-1, vmax=1, cmap=board_layout_cmap)
        fig.tight_layout()

        plt.pause(0.1)
        print("Player {} took action {}".format(1 if is_initiators_turn else 2, id2action(action.squeeze().item())))
        is_terminal = board.take(*id2action(action_id), side=1 if is_initiators_turn else 2)
        print(board.as_str())
        is_initiators_turn = not is_initiators_turn
    print("Result", BoardState(board.occupancy))
    plt.show()


def interactive_test(model_builder, output_dir, device, input_step_size):
    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    board_layout_cmap = "RdYlBu"
    fig = plt.figure(figsize=(15, 10))
    ax_next_player = fig.add_subplot(521)
    ax_board_layout = fig.add_subplot(523)
    ax_logits = fig.add_subplot(522)
    ax_probs = fig.add_subplot(524)
    ax_action = fig.add_subplot(525)
    ax_probs_valid = fig.add_subplot(526)
    ax_entropy = fig.add_subplot(514)
    ax_values = fig.add_subplot(515)
    ax_next_player.set_title("Next Player")
    ax_board_layout.set_title("Board Layout")
    ax_logits.set_title("Logits from Policy Network")
    ax_probs.set_title(r"$\pi(\mathbf{a}|\mathbf{s})$ (w/o Valid Action Mask) from Policy Network")
    ax_probs_valid.set_title(r"$\pi(\mathbf{a}|\mathbf{s})$ (with Valid Action Mask) from Policy Network")
    ax_action.set_title("Action Taken")
    nn = model_builder()
    ckpt_n_episode, path = get_last_checkpoint(output_dir)
    checkpoint = torch.load(path, map_location=device)
    nn.load_state_dict(checkpoint["edge"])
    nn.to(device)
    nn.eval()
    nn1 = nn2 = nn
    is_terminal = False
    is_initiators_turn = True
    board = UltimateTicTacToe.create_initial_board()
    action_history = []
    board_layout_history = [board.get_state() for _ in range(input_step_size - 1)]
    valid_policy_entropy_history, valid_policy_entropy_history_player1, valid_policy_entropy_history_player2 = [], [], []
    policy_entropy_history = []
    value_history = []
    play_as_initiator = input("Play as initiator?")
    if play_as_initiator.lower() in {"1", "y", "yes", "true"}:
        play_as_initiator = True
    elif play_as_initiator.lower() in {"0", "n", "no", "false"}:
        play_as_initiator = False
    else:
        raise ValueError("Unrecognized value: {}".format(play_as_initiator))
    while not is_terminal:
        is_agents_turn = play_as_initiator ^ is_initiators_turn
        if is_initiators_turn:
            nn = nn1
        else:
            nn = nn2
        board_layout = board.get_state()
        board_layout_history.append(board_layout)
        batch_board_layout = torch.from_numpy(np.stack(board_layout_history[-input_step_size:], axis=-1)).to(dtype=torch.int64, device=device).unsqueeze(0)
        valid_actions = board.valid_actions
        valid_actions = torch.tensor(sorted(map(lambda x: action2id(*x), valid_actions)), device=device)
        valid_action_mask = torch.tensor([idx in set(map(lambda x: action2id(*x), board.valid_actions)) for idx in range(81)], device=device)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        print("Board :\n{}\nValue: {}".format(board.as_str(), values.squeeze().item()))
        valid_policy_logits = policy_logits[:, valid_actions]
        # p_action_val = policy_logits[0].tolist()
        # create a categorical distribution over the list of probabilities of actions
        valid_probs = torch.softmax(valid_policy_logits, dim=-1)
        # print(batch_board_layout.squeeze(0), policy_logits.view(-1, 9, 9), "valid_action_logits", valid_policy_logits, "valid_action_probs", valid_probs)
        ax_next_player.imshow(np.full([9, 9], 1 if is_initiators_turn else -1), vmin=-1, vmax=1, cmap=board_layout_cmap)
        ax_next_player.xaxis.set_tick_params(length=0)
        ax_next_player.yaxis.set_tick_params(length=0)
        ax_board_layout.imshow(np.where(board_layout != 2, board_layout, np.full_like(board_layout, -1)), vmin=-1, vmax=1, cmap=board_layout_cmap)
        ax_logits.imshow(policy_logits.view(9, 9).numpy())
        ax_probs.imshow(torch.softmax(policy_logits, dim=-1).view(9, 9).numpy(), vmin=0, vmax=1)
        ax_probs_valid.imshow(torch.softmax(torch.where(valid_action_mask, policy_logits, torch.full_like(policy_logits, -50000)), dim=-1).view(9, 9), vmin=0, vmax=1)
        m1 = torch.distributions.Categorical(valid_probs)

        # and sample an action using the distribution
        action = m1.sample()

        ax_entropy.cla()
        ax_entropy.set_title(r"Entropy of $\pi(\mathbf{a}|\mathbf{s})$")
        entropy_raw = torch.distributions.Categorical(torch.softmax(policy_logits, dim=-1)).entropy().item()
        entropy_valid = m1.entropy()
        policy_entropy_history.append(entropy_raw)
        valid_policy_entropy_history.append(entropy_valid)
        if is_initiators_turn:
            valid_policy_entropy_history_player1.append(entropy_valid)
        else:
            valid_policy_entropy_history_player2.append(entropy_valid)
        ax_entropy.scatter(np.arange(0, 2 * len(valid_policy_entropy_history_player1), 2), valid_policy_entropy_history_player1, color="c", label="Player 1")
        ax_entropy.scatter(np.arange(1, 1 + 2 * len(valid_policy_entropy_history_player2), 2), valid_policy_entropy_history_player2, color="r", label="Player 2")
        ax_entropy.plot(policy_entropy_history, label="w/o valid action mask")
        ax_entropy.plot(valid_policy_entropy_history, label="with valid action mask")

        ax_entropy.legend()
        value_history.append(values.squeeze().item())
        ax_values.cla()
        ax_values.set_title(r"$v(\mathbf{s})$ for Player 1")
        ax_values.plot(value_history)

        action = valid_actions[action]
        action_id = action.squeeze().item()
        action_history.append(action_id)
        if is_agents_turn:
            ax_action.imshow(np.where(np.arange(81).reshape(9, 9) == action_id, 1 if is_initiators_turn else -1, 0), vmin=-1, vmax=1, cmap=board_layout_cmap)
        fig.tight_layout()

        plt.pause(0.1)
        if is_agents_turn:
            print("Player {} took action {}".format(1 if is_initiators_turn else 2, id2action(action.squeeze().item())))
            is_terminal = board.take(*id2action(action_id), side=1 if is_initiators_turn else 2)
        else:
            print("Agent suggested action {}".format(id2action(action.squeeze().item())))
            while True:
                try:
                    action = input("Enter your action:")
                    action = tuple(map(int, action))
                    print(action, board.valid_actions, action not in board.valid_actions)
                    if action not in board.valid_actions:
                        print("Action {} is beyond valid actions: {}".format(board.valid_actions))
                        continue
                    else:
                        break
                except Exception as e:
                    print(e)
            is_terminal = board.take(*action, side=1 if is_initiators_turn else 2)
        print(board.as_str())
        is_initiators_turn = not is_initiators_turn
    print("Result", BoardState(board.occupancy))
    plt.show()


def eval_agent(nn1: torch.nn.Module, nn1_input_step_size, nn2: torch.nn.Module, nn2_input_step_size, device):
    is_terminal = False
    is_initiators_turn = True
    board = UltimateTicTacToe.create_initial_board()
    n_step = max(nn1_input_step_size, nn2_input_step_size)
    state_buff = [board.get_state() for _ in range(n_step - 1)]
    action_history = []
    policy_entropy_history = []
    while not is_terminal:
        if is_initiators_turn:
            nn = nn1
        else:
            nn = nn2
        board_layout = board.get_state()
        state_buff.append(board_layout)
        state_buff = state_buff[-n_step:]
        batch_board_layout = torch.from_numpy(np.stack(state_buff[-(nn1_input_step_size if is_initiators_turn else nn2_input_step_size):], axis=-1)).to(dtype=torch.int64, device=device).unsqueeze(0)
        valid_actions = board.valid_actions
        valid_actions = torch.tensor(sorted(map(lambda x: action2id(*x), valid_actions)), device=device)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        # print("Board :\n{}\nValue: {}".format(board.as_str(), values.squeeze().item()))
        valid_policy_logits = policy_logits[:, valid_actions]
        # p_action_val = policy_logits[0].tolist()
        # create a categorical distribution over the list of probabilities of actions
        valid_probs = torch.softmax(valid_policy_logits, dim=-1)
        # print(batch_board_layout, policy_logits.view(-1, 9, 9), valid_probs)
        m1 = torch.distributions.Categorical(valid_probs)
        policy_entropy_history.append(m1.entropy().item())
        # and sample an action using the distribution
        action = m1.sample()
        action = valid_actions[action]
        action_id = action.squeeze().item()
        action_history.append(action_id)
        # print("Player {} took action {}".format(1 if is_initiators_turn else 2, id2action(action.squeeze().item())))
        is_terminal = board.take(*id2action(action_id), side=1 if is_initiators_turn else 2)
        # print(board.as_str())
        is_initiators_turn = not is_initiators_turn
    return {"board_occupancy": board.occupancy, "actions": action_history, "policy_entropy_history": policy_entropy_history}


def eval_network(nn1: torch.nn.Module,
                 nn1_input_step_size: int,
                 nn2: torch.nn.Module,
                 nn2_input_step_size: int,
                 device,
                 n_episode=10000):
    n_p1_wins, n_p2_wins, draw, p1_acc_reward = 0, 0, 0, 0
    for _ in range(n_episode):
        result = eval_agent(nn1, nn1_input_step_size, nn2, nn2_input_step_size, device)
        board_occupancy = result["board_occupancy"]
        # Calculate reward
        if board_occupancy == BoardState.OCCUPIED_BY_PLAYER1:
            p1_acc_reward += 1
            n_p1_wins += 1
        elif board_occupancy == BoardState.OCCUPIED_BY_PLAYER2:
            p1_acc_reward += -1
            n_p2_wins += 1
        else:
            p1_acc_reward += 0
            draw += 1
    return {"n_p1_wins": n_p1_wins, "n_p2_wins": n_p2_wins, "n_draw": draw, "expected_reward": p1_acc_reward / n_episode}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--do-visualize", action="store_true")
    parser.add_argument("--do-interactive-eval", action="store_true")
    args = parser.parse_args()
    main(args)
