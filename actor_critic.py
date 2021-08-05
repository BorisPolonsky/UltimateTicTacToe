from ultimate_tic_tac_toe.game_board import UltimateTicTacToe, BoardState
import torch
import torch.nn.functional as F
from typing import Tuple, Callable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import glob
import re
from collections import namedtuple


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


class UniformRandomSamplingNN(torch.nn.Module):
    """An agent equivalent to uniform random sampling"""
    def __init__(self):
        super().__init__()

    def forward(self, board_layout: torch.Tensor, next_player: torch.Tensor):
        batch_size = board_layout.size(0)
        device = board_layout.device
        dtype = torch.float32
        policy = torch.zeros([batch_size, 81], dtype=dtype, device=device)
        value = torch.zeros([batch_size], dtype=dtype, device=device)
        return policy, value


class Env:
    """OpenAI-Gym styled environment wrapper"""
    def __init__(self, init_board_fn, n_step: int):
        if not n_step > 0:
            raise ValueError("n_step must be a positive integer, got {}".format(n_step))
        self.n_step = n_step
        self.init_board_fn: Callable[[], UltimateTicTacToe] = init_board_fn
        self.board: UltimateTicTacToe = None
        self.next_player_id = 0
        self.buff = []
        self.reset()

    def reset(self):
        self.board = self.init_board_fn()
        self.next_player_id = 0
        self.buff = [self.board.get_state() for _ in range(self.n_step)]
        board_layout_history = self._get_historical_board_layout()
        valid_actions = self._get_valid_actions()
        obs = board_layout_history, valid_actions, self.next_player_id
        return obs

    def _get_valid_actions(self):
        valid_actions = sorted(map(lambda x: action2id(*x), self.board.valid_actions))
        return valid_actions

    def _get_historical_board_layout(self):
        board_layout_history = np.stack(self.buff[-self.n_step:], axis=-1)
        return board_layout_history

    def step(self, action: Tuple[int, int]):
        """
        :param action:
        :return: Tuple of (observation, reward, done, info)
        - observation: Tuple of [board_layout_history, valid_actions, next_player_id
            - board_layout_history: shape [9, 9, n_step], dtype: int
            - valid_actions: List of int (e.g. [0, 1, ..., 80])
            - next_player_id: int 0 (initiator) or 1
        - reward: float. 0. if not done or draw, 1. if initiator wins, -1 if initiator loses.
        - info: dict
        """
        action_id, next_player_id = action
        board = self.board
        reward = 0.
        info = {}
        side = next_player_id + 1
        done = board.take(*id2action(action_id), side=side)
        if done:
            if board.occupancy == BoardState.OCCUPIED_BY_PLAYER1:
                reward = 1.
            else:
                reward = -1.
            info["board_occupancy"] = board.occupancy
        self.buff.append(self.board.get_state())
        del self.buff[:-self.n_step]
        board_layout_history = self._get_historical_board_layout()
        valid_actions = self._get_valid_actions()
        self.next_player_id = 0 if self.next_player_id != 0 else 1
        observation = board_layout_history, valid_actions, self.next_player_id
        return observation, reward, done, info


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


class DataAugmentation:
    def __init__(self):
        self._state_index_mapping = self._get_state_index_mapping()

    @classmethod
    def rotate_90_cw(cls, row_block: int, column_block: int, row_slot: int, column_slot: int):
        return column_block, 2 - row_block, column_slot, 2 - row_slot

    @classmethod
    def rotate_180_cw(cls, row_block: int, column_block: int, row_slot: int, column_slot: int):
        return 2 - row_block, 2 - column_block, 2 - row_slot, 2 - column_slot

    @classmethod
    def rotate_270_cw(cls, row_block: int, column_block: int, row_slot: int, column_slot: int):
        return 2 - column_block, row_block, 2 - column_slot, row_slot

    @classmethod
    def flip_horizontal(cls, row_block: int, column_block: int, row_slot: int, column_slot: int):
        return row_block, 2 - column_block, row_slot, 2 - column_slot

    def _get_state_index_mapping(self):
        action_id = 0
        out = {op_name: {new_action_id: 0} for op_name, new_action_id in self.action_augmentation(action_id).items()}
        for action_id in range(1, 81):
            for op_name, new_action_id in self.action_augmentation(action_id).items():
                mapping = out[op_name]
                mapping[new_action_id] = action_id
        for op_name in out:
            mapping = out[op_name]
            out[op_name] = np.array([mapping[action_id] for action_id in range(81)])
        return out

    def action_augmentation(self, action_id: int):
        action = id2action(action_id)
        flipped_action = self.flip_horizontal(*action)
        new_actions = {
            "rotate_90_clockwise": self.rotate_90_cw(*action),
            "rotate_180_clockwise": self.rotate_180_cw(*action),
            "rotate_270_clockwise": self.rotate_270_cw(*action),
            "flip_horizontal": flipped_action,
            "flip_horizontal_rotate_90_clockwise": self.rotate_90_cw(*flipped_action),
            "flip_horizontal_rotate_180_clockwise": self.rotate_180_cw(*flipped_action),
            "flip_horizontal_rotate_270_clockwise": self.rotate_270_cw(*flipped_action)
        }
        return {k: action2id(*v) for k, v in new_actions.items()}

    def action_sequence_augmentation(self, action_sequence):
        out = {op_name: [] for op_name in self._state_index_mapping.keys()}
        for action_id in action_sequence:
            new_action_ids = self.action_augmentation(action_id)
            for op_name in new_action_ids:
                out[op_name].append(new_action_ids[op_name])
        return out

    def state_sequence_augmentation(self, state_sequence):
        out = {op_name: [] for op_name in self._state_index_mapping.keys()}
        for state in state_sequence:
            new_states = self.state_augmentation(state)
            for op_name in new_states:
                out[op_name].append(new_states[op_name])
        return out

    def state_augmentation(self, state):
        *dims, h, w, n_steps = list(state.shape)
        flatten_view = state.reshape(dims + [81, n_steps])
        out = {}
        for action_name, indices in self._state_index_mapping.items():
            out[action_name] = flatten_view[..., indices, :].reshape(dims + [9, 9, n_steps])
        return out

    def valid_action_augmentation(self, valid_action_set_sequence, sort=False):
        out = {op_name: [] for op_name in self._state_index_mapping.keys()}
        for valid_action_ids_at_t in valid_action_set_sequence:
            for op_name, new_valid_action_ids_at_t in self.action_sequence_augmentation(valid_action_ids_at_t).items():
                if sort:
                    new_valid_action_ids_at_t = sorted(new_valid_action_ids_at_t)
                out[op_name].append(new_valid_action_ids_at_t)
        return out


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
    input_step_size = 2
    model_builder = get_model_builder(input_step_size)
    rules = {
        "sovereignty_upon_draw": "none"
    }

    def board_init_fn():
        return UltimateTicTacToe.create_initial_board(**rules)

    env = Env(board_init_fn, n_step=input_step_size)
    if args.do_train:
        use_gpu = True
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        train_fn(env, model_builder, output_dir, device, input_step_size)

    if args.do_eval:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        eval_fn(env, model_builder, output_dir, device, n_episode=500)
    if args.do_visualize:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        visualize(env, model_builder, output_dir, device)
    if args.do_interactive_eval:
        use_gpu = False
        device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        interactive_test(env, model_builder, output_dir, device)


def train_fn(env, model_builder, output_dir, device, n_input_step: int):
    nn = model_builder()
    nn_current_best = model_builder()
    Episode = namedtuple("Episode", ("state_history", "action_history", "reward_to_go", "valid_action_history"))
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
    alpha_l2_regularization = 1e-2
    weight_p, bias_p = [], []
    weight_name, bias_name = [], []
    for name, param in nn.named_parameters():
        if name.endswith("bias"):
            bias_p.append(param)
            bias_name.append(name)
        else:
            weight_p.append(param)
            weight_name.append(name)
    print("The following parameters will be optimized WITH decay")
    print("\n".join(weight_name))
    print("The following parameters will be optimized WITHOUT decay")
    print("\n".join(bias_name))
    param_optim_config = [
            {'params': weight_p, 'weight_decay': alpha_l2_regularization},
            {'params': bias_p, 'weight_decay': 0}
        ]
    # optimizer = torch.optim.SGD(param_optim_config, lr=2e-4)
    optimizer = torch.optim.AdamW(param_optim_config, lr=2e-5, weight_decay=alpha_l2_regularization)
    # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)
    del weight_p, bias_p, weight_name, bias_name, param_optim_config

    alpha_ema = 0.3
    alpha_value_loss = 10
    beta_policy_regularization = 2e-3
    log_every = 1000
    eval_every = 5000
    save_checkpoints_per_n_episode = 10000  # How often to save state dict
    save_checkpoints_per_n_seconds = 1 * 3600
    data_augmentation = DataAugmentation()
    augment_data = True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o755)
    with SummaryWriter(os.path.join(output_dir, "log")) as writer:
        tic = time.time()
        reward_ema = ExponentialMovingAverage(alpha_ema)
        num_episode = prev_save_idx = ckpt_n_episode
        n_episode_current_generation = prev_log_idx = prev_eval_idx = 0
        generation = 0
        batch_size = 128
        batch = []
        for batch_i in range(num_batch):
            while len(batch) < batch_size:
                # Simulate one episode
                state_history = []
                action_history = []
                valid_action_history = []
                reward_history = []
                done = False
                is_initiators_turn = True
                observation = env.reset()
                while not done:
                    board_layout, valid_actions, _ = observation
                    state_history.append(board_layout)
                    batch_board_layout = torch.from_numpy(board_layout).to(dtype=torch.int64, device=device).unsqueeze(0)
                    batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
                    valid_action_history.append(valid_actions)
                    valid_actions = torch.tensor(valid_actions, device=device)
                    with torch.no_grad():
                        policy_logits, values = nn(batch_board_layout, batch_next_player)
                    valid_policy_logits = policy_logits[:, valid_actions]
                    # create a categorical distribution over the list of probabilities of actions
                    m1 = torch.distributions.Categorical(logits=valid_policy_logits)
                    # and sample an action using the distribution
                    action = m1.sample()
                    action = valid_actions[action].squeeze().item()
                    observation, reward, done, info = env.step((action, 0 if is_initiators_turn else 1))
                    reward_history.append(reward)
                    action_history.append(action)
                    is_initiators_turn = not is_initiators_turn
                    del valid_policy_logits, policy_logits, values, action, valid_actions, batch_board_layout, batch_next_player
                # Calculate reward
                reward_ema.update(reward)
                reward_to_go = []
                g_t = 0
                for reward in reward_history[::-1]:
                    g_t = reward + g_t * gamma
                    reward_to_go.insert(0, g_t)
                del g_t, reward_history
                batch.append(Episode(state_history=np.stack(state_history, axis=0),  # [L_episode, 9, 9, n_history_step]
                                     action_history=np.array(action_history),
                                     reward_to_go=np.array(reward_to_go, dtype=np.float32),
                                     valid_action_history=valid_action_history))
                # Data augmentation
                if augment_data:
                    action_history_augment = data_augmentation.action_sequence_augmentation(action_history)
                    state_history_augment = data_augmentation.state_sequence_augmentation(state_history)
                    valid_action_history_augment = data_augmentation.valid_action_augmentation(valid_action_history, sort=True)
                    for op_name in action_history_augment:
                        action_history = action_history_augment[op_name]
                        state_history = state_history_augment[op_name]
                        valid_action_history = valid_action_history_augment[op_name]
                        batch.append(Episode(state_history=np.stack(state_history, axis=0),  # [L_episode, 9, 9, n_history_step]
                                             action_history=np.array(action_history),
                                             reward_to_go=np.array(reward_to_go, dtype=np.float32),
                                             valid_action_history=valid_action_history))
                del reward_to_go

            # Form a batch
            batch_input_state = []
            batch_action = []
            batch_reward_to_go = []  # Reward to go w.r.t Player 1
            batch_action_num = []
            batch_valid_action_ids = []
            if len(batch) > batch_size:
                batch = batch[:batch_size]
            max_l_episode = 0
            for item in batch:
                action_num = len(item.action_history)
                max_l_episode = max(max_l_episode, action_num)
                batch_action_num.append(action_num)
                batch_input_state.append(torch.from_numpy(item.state_history))
                batch_action.append(torch.from_numpy(item.action_history))
                batch_reward_to_go.append(torch.from_numpy(item.reward_to_go))
                batch_valid_action_ids.append(item.valid_action_history)
            batch = []
            # [batch_size]
            batch_action_num = torch.tensor(batch_action_num)
            # [L]
            batch_action_flatten = torch.cat(batch_action)
            # [L, 9, 9, n_history_step]
            batch_input_state_flatten = torch.cat(batch_input_state, dim=0)
            # # [L]
            batch_reward_to_go_flatten = torch.cat(batch_reward_to_go).detach()
            # Get player_id
            batch_player_id_flatten = (torch.arange(max_l_episode) % 2).repeat(batch_size)
            batch_time_step_flatten = torch.arange(max_l_episode).repeat(batch_size)
            batch_time_step_limit_flatten = batch_action_num.unsqueeze(-1).repeat(1, max_l_episode).flatten()
            batch_valid_input_mask = batch_time_step_flatten < batch_time_step_limit_flatten
            batch_player_id_flatten = batch_player_id_flatten[batch_valid_input_mask].detach()
            del batch_time_step_flatten, batch_time_step_limit_flatten, batch_valid_input_mask
            # Get valid action mask
            sparse_indices = []
            idx_dim0 = 0
            for valid_action_history in batch_valid_action_ids:
                for valid_actions_at_t in valid_action_history:
                    for idx_dim1 in valid_actions_at_t:
                        sparse_indices.append([idx_dim0, idx_dim1])
                    idx_dim0 += 1
            assert idx_dim0 == batch_action_flatten.size(0)
            sparse_values = torch.tensor([1] * len(sparse_indices), dtype=torch.int32)
            sparse_indices = torch.tensor(sparse_indices).T  # [2, L]
            valid_action_mask_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, [idx_dim0, 81])
            valid_action_mask = (valid_action_mask_sparse.to_dense() == 1)
            del sparse_indices, sparse_values, idx_dim0, idx_dim1, valid_action_mask_sparse
            # Move tensors to devices
            batch_input_state_flatten = batch_input_state_flatten.to(device=device)
            batch_player_id_flatten = batch_player_id_flatten.to(device=device)
            batch_reward_to_go_flatten = batch_reward_to_go_flatten.to(device=device)
            batch_action_flatten = batch_action_flatten.to(device=device)
            valid_action_mask = valid_action_mask.to(device=device)
            # unmasked_policy_logits: [L, 81], values: [L]
            unmasked_policy_logits, values = nn(batch_input_state_flatten, batch_player_id_flatten)
            invalid_action_logits = torch.empty_like(unmasked_policy_logits)
            torch.fill_(invalid_action_logits, -50000)
            # masked_policy_logits: [L, 81], values: [L]
            masked_policy_logits = torch.where(valid_action_mask, unmasked_policy_logits, invalid_action_logits)
            num_episode += batch_size
            n_episode_current_generation += batch_size

            m_unmasked = torch.distributions.Categorical(logits=unmasked_policy_logits)
            m_masked = torch.distributions.Categorical(logits=masked_policy_logits)
            action_log_probs = m_masked.log_prob(batch_action_flatten)
            # action_log_probs = m_unmasked.log_prob(batch_action_flatten)
            unmasked_policy_entropy = m_unmasked.entropy()
            masked_policy_entropy = m_masked.entropy()
            p1_mask = (batch_player_id_flatten == 0)
            p2_mask = ~p1_mask
            n_action_p1 = torch.sum(p1_mask.to(torch.int64)).item()
            n_action_p2 = torch.sum(p2_mask.to(torch.int64)).item()
            log_probs_p1 = action_log_probs[p1_mask]
            log_probs_p2 = action_log_probs[p2_mask]
            values_no_grad = values.detach()
            advantage_p1 = batch_reward_to_go_flatten[p1_mask] - values_no_grad[p1_mask]
            advantage_p2 = -batch_reward_to_go_flatten[p2_mask] + values_no_grad[p2_mask]  # value_p1 == -value_p2
            p1_policy_loss_unnormalized = -torch.sum(torch.mul(log_probs_p1, advantage_p1))
            p2_policy_loss_unnormalized = -torch.sum(torch.mul(log_probs_p2, advantage_p2))
            p_loss = (p1_policy_loss_unnormalized + p2_policy_loss_unnormalized) / batch_size
            v_loss = F.smooth_l1_loss(values, batch_reward_to_go_flatten, reduction="mean", beta=0.05)
            entropy_regularization_term = -torch.sum(unmasked_policy_entropy) / batch_size
            loss = p_loss + alpha_value_loss * v_loss + beta_policy_regularization * entropy_regularization_term
            # reset gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            toc = time.time()

            if n_episode_current_generation - prev_log_idx > log_every:
                prev_log_idx = n_episode_current_generation
                writer.add_scalar("player1_policy_loss", (p1_policy_loss_unnormalized / batch_size).item(), num_episode)
                writer.add_scalar("player2_policy_loss", (p2_policy_loss_unnormalized / batch_size).item(), num_episode)
                writer.add_scalar("policy_loss", p_loss.item(), num_episode)
                writer.add_scalar("value_loss", v_loss.item(), num_episode)
                writer.add_scalar("total_loss", loss.item(), num_episode)
                writer.add_histogram("unmasked_policy_entropy", unmasked_policy_entropy, num_episode)  # max: 4.3944
                writer.add_histogram("policy_entropy", masked_policy_entropy, num_episode)
                writer.add_histogram("unmasked_policy_entropy_p1", unmasked_policy_entropy[p1_mask], num_episode)
                writer.add_histogram("unmasked_policy_entropy_p2", unmasked_policy_entropy[p2_mask], num_episode)
                writer.add_histogram("policy_entropy_p1", masked_policy_entropy[p1_mask], num_episode)
                writer.add_histogram("policy_entropy_p2", masked_policy_entropy[p2_mask], num_episode)
                writer.add_scalar("value_explained_variance", 1 - (values_no_grad.detach() - batch_reward_to_go_flatten).var() / (1e-9 + batch_reward_to_go_flatten.var()), num_episode)
                print("loss", loss.item(), "policy_loss", p_loss.item(), "p1_policy_loss", p1_policy_loss_unnormalized.item(),  "p2_policy_loss", p2_policy_loss_unnormalized.item(), "v_loss", v_loss.item())
                print("Reward for Player1 (EMA):", reward_ema.result())
            if (n_episode_current_generation - prev_eval_idx) > eval_every:
                prev_eval_idx = n_episode_current_generation
                n_eval_per_role = 400
                nn_as_p1_result = eval_network(env, nn, nn_current_best, device, n_eval_per_role)
                nn_as_p2_result = eval_network(env, nn_current_best, nn, device, n_eval_per_role)
                nn_win_rate = (nn_as_p1_result["n_p1_wins"] + nn_as_p2_result["n_p2_wins"]) / (2 * n_eval_per_role)
                nn_as_p1_win_rate = nn_as_p1_result["n_p1_wins"] / n_eval_per_role
                nn_as_p2_win_rate = nn_as_p2_result["n_p2_wins"] / n_eval_per_role
                current_best_nn_win_rate = (nn_as_p1_result["n_p2_wins"] + nn_as_p2_result["n_p1_wins"]) / (2 * n_eval_per_role)
                smoothed_nn_win_rate = (nn_as_p1_result["n_p1_wins"] + nn_as_p2_result["n_p2_wins"] + 0.5 * (nn_as_p1_result["n_draw"] + nn_as_p2_result["n_draw"])) / (2 * n_eval_per_role)
                print("win_rate_as_p1", nn_as_p1_win_rate, "win_rate_as_p2", nn_as_p2_win_rate, "smoothed_nn_win_rate", smoothed_nn_win_rate)
                # accept_new_network = (current_best_nn_win_rate == 0 and nn_win_rate > 0) or smoothed_nn_win_rate > 0.55 # nn_win_rate / current_best_nn_win_rate > 1.2
                accept_new_network = (nn_as_p1_win_rate > 0.55) and (nn_as_p2_win_rate > 0.55)
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

            if accept_new_network or (toc - tic) > save_checkpoints_per_n_seconds or (num_episode - prev_save_idx > save_checkpoints_per_n_episode):
                tic = time.time()
                checkpoint = {"current_best": nn_current_best.state_dict(), "edge": nn.state_dict()}
                torch.save(checkpoint, os.path.join(output_dir, "checkpoint-{}.pt".format(num_episode)))
                prev_save_idx = num_episode
                del checkpoint


def eval_fn(env, model_builder, output_dir, device, n_episode):
    nn = model_builder()
    ckpt_n_episode, path = get_last_checkpoint(output_dir)
    checkpoint = torch.load(path, map_location=device)
    nn.load_state_dict(checkpoint["current_best"])
    nn.to(device)
    nn.eval()
    print("Evaluating: Self-play")
    result = eval_network(env, nn, nn, device, n_episode=n_episode)
    print(result)
    nn_random_sampling = UniformRandomSamplingNN()
    nn_random_sampling.to(device)
    print("Evaluating: Player1(nn) vs. Player2(Uniform-Random-Sampling)")
    result = eval_network(env, nn, nn_random_sampling, device, n_episode=n_episode)
    print(result)
    print("Evaluating: Player1(Uniform-Random-Sampling) vs. Player2(nn)")
    result = eval_network(env, nn_random_sampling, nn, device, n_episode=n_episode)
    print(result)


def visualize(env, model_builder, output_dir, device):
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
    nn.load_state_dict(checkpoint["current_best"])
    nn.to(device)
    nn.eval()
    nn1 = nn2 = nn
    done = False
    is_initiators_turn = True
    observation = env.reset()
    action_history = []
    valid_policy_entropy_history, valid_policy_entropy_history_player1, valid_policy_entropy_history_player2 = [], [], []
    policy_entropy_history = []
    value_history = []
    while not done:
        if is_initiators_turn:
            nn = nn1
        else:
            nn = nn2
        board_layout_history, valid_actions, _ = observation
        board_layout = env.board.get_state()
        batch_board_layout = torch.from_numpy(board_layout_history).to(dtype=torch.int64, device=device).unsqueeze(0)
        valid_action_mask = torch.tensor([idx in set(valid_actions) for idx in range(81)], device=device)
        valid_actions = torch.tensor(valid_actions, device=device)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        print("Board :\n{}\nValue: {}".format(env.board.as_str(), values.squeeze().item()))
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
        m1 = torch.distributions.Categorical(logits=valid_policy_logits)
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
        observation, reward, done, info = env.step((action_id, 0 if is_initiators_turn else 1))
        print(env.board.as_str())
        is_initiators_turn = not is_initiators_turn
    print("Result", BoardState(info["board_occupancy"]))
    plt.show()


def interactive_test(env, model_builder, output_dir, device):
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
    nn.load_state_dict(checkpoint["current_best"])
    nn.to(device)
    nn.eval()
    nn1 = nn2 = nn
    done = False
    is_initiators_turn = True
    observation = env.reset()
    action_history = []
    valid_policy_entropy_history, valid_policy_entropy_history_player1, valid_policy_entropy_history_player2 = [], [], []
    policy_entropy_history = []
    value_history = []
    play_as_initiator = input("Play as initiator?")
    # board_as_str_config = {"token1": "\u2B55", "token2": "\u274C"}
    board_as_str_config = {}
    if play_as_initiator.lower() in {"1", "y", "yes", "true"}:
        play_as_initiator = True
    elif play_as_initiator.lower() in {"0", "n", "no", "false"}:
        play_as_initiator = False
    while not done:
        is_agents_turn = play_as_initiator ^ is_initiators_turn
        board_layout_history, valid_actions, _ = observation
        board_layout = env.board.get_state()
        batch_board_layout = torch.from_numpy(board_layout_history).to(dtype=torch.int64, device=device).unsqueeze(0)
        valid_action_mask = torch.tensor([idx in set(valid_actions) for idx in range(81)], device=device)
        valid_actions = torch.tensor(valid_actions, device=device)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        print("Board :\n{}\nValue: {}".format(env.board.as_str(**board_as_str_config), values.squeeze().item()))
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
        m1 = torch.distributions.Categorical(logits=valid_policy_logits)
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
        if is_agents_turn:
            print("Player {} took action {}".format(1 if is_initiators_turn else 2, id2action(action_id)))
        else:
            print("Agent suggested action {}".format(id2action(action.squeeze().item())))
            while True:
                try:
                    action = input("Enter your action:")
                    action = tuple(map(int, action))
                    action_id = action2id(*action)
                    if action_id not in valid_actions:
                        print("Action {} is beyond valid actions: {}".format(action, env.board.valid_actions))
                        continue
                    else:
                        break
                except Exception as e:
                    print(e)
        observation, reward, done, info = env.step((action_id, 0 if is_initiators_turn else 1))
        print(env.board.as_str(**board_as_str_config))
        is_initiators_turn = not is_initiators_turn
    print("Result", BoardState(info["board_occupancy"]))
    plt.show()


def eval_agent(env, nn1: torch.nn.Module, nn2: torch.nn.Module, device):
    done = False
    is_initiators_turn = True
    action_history = []
    policy_entropy_history = []
    observation = env.reset()
    board_layout_history, valid_actions, _ = observation
    while not done:
        if is_initiators_turn:
            nn = nn1
        else:
            nn = nn2
        batch_board_layout = torch.from_numpy(board_layout_history).to(dtype=torch.int64, device=device).unsqueeze(0)
        batch_next_player = torch.tensor([0 if is_initiators_turn else 1], dtype=torch.int64, device=device)
        with torch.no_grad():
            policy_logits, values = nn(batch_board_layout, batch_next_player)
        # print("Board :\n{}\nValue: {}".format(board.as_str(), values.squeeze().item()))
        valid_policy_logits = policy_logits[:, valid_actions]
        # p_action_val = policy_logits[0].tolist()
        # create a categorical distribution over the list of probabilities of actions
        m1 = torch.distributions.Categorical(logits=valid_policy_logits)
        policy_entropy_history.append(m1.entropy().item())
        # and sample an action using the distribution
        action = m1.sample()
        action_id = valid_actions[action]
        action_history.append(action_id)
        observation, reward, done, info = env.step((action_id, 0 if is_initiators_turn else 1))
        board_layout_history, valid_actions, _ = observation
        is_initiators_turn = not is_initiators_turn
    return {"board_occupancy": info["board_occupancy"], "actions": action_history, "policy_entropy_history": policy_entropy_history}


def eval_network(env,
                 nn1: torch.nn.Module,
                 nn2: torch.nn.Module,
                 device,
                 n_episode=10000):
    n_p1_wins, n_p2_wins, draw, p1_acc_reward = 0, 0, 0, 0
    for _ in range(n_episode):
        result = eval_agent(env, nn1, nn2, device)
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
