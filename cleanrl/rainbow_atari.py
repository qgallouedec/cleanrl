# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_ataripy
import argparse
import math
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--n-atoms", type=int, default=51,
        help="the number of atoms")
    parser.add_argument("--v-min", type=float, default=-10,
        help="the return lower bound")
    parser.add_argument("--v-max", type=float, default=10,
        help="the return upper bound")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=10000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Capacity of the sum tree (number of leaves)
        self.tree = [0] * (2 * capacity)  # Binary tree representation
        self.max_priority = 1.0  # Initial max priority for new experiences

    def update(self, index, priority=None):
        if priority is None:
            priority = self.max_priority
        tree_idx = index + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self.max_priority = max(self.max_priority, priority)

    def _propagate(self, idx, change):
        parent = idx // 2
        while parent != 0:
            self.tree[parent] += change
            parent = parent // 2

    def total(self):
        return self.tree[1]  # The root of the tree holds the total sum

    def get(self, s):
        idx = 1
        while idx < self.capacity:  # Keep moving down the tree to find the index
            left = 2 * idx
            right = left + 1
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx - self.capacity


class PrioritizedReplayBuffer:
    def __init__(self, size, device, alpha=0.5, beta_0=0.4, n_step=3, gamma=0.99):
        self.size = size
        self.device = device
        self.alpha = alpha
        self.beta_0 = beta_0
        self.update_beta(0.0)
        self.n_step = n_step
        self.gamma = gamma

        self.next_index = 0
        self.sum_tree = SumTree(size)
        self.observations = np.zeros((self.size, 4, 84, 84), dtype=np.uint8)
        self.next_observations = np.zeros((self.size, 4, 84, 84), dtype=np.uint8)
        self.actions = np.zeros((self.size, 1), dtype=np.int64)
        self.rewards = np.zeros((self.size, 1), dtype=np.float32)
        self.dones = np.zeros((self.size, 1), dtype=bool)

        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, obs, next_obs, actions, rewards, dones, infos):
        self.n_step_buffer.append((obs[0], next_obs[0], actions[0], rewards[0], dones[0], infos))

        if len(self.n_step_buffer) < self.n_step and not dones[0]:
            return

        # Compute n-step return and the first state and action
        rewards = [self.n_step_buffer[i][3] for i in range(len(self.n_step_buffer))]
        n_step_return = sum([r * (self.gamma**i) for i, r in enumerate(rewards)])
        obs, _, action, _, _, _ = self.n_step_buffer[0]
        _, next_obs, _, _, done, _ = self.n_step_buffer[-1]

        # Store the n-step transition
        self.observations[self.next_index] = obs
        self.next_observations[self.next_index] = next_obs
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = n_step_return
        self.dones[self.next_index] = done

        # Get the max priority in the tree and set the new transition with max priority
        self.sum_tree.update(self.next_index)
        self.next_index = (self.next_index + 1) % self.size

        if dones[0]:
            self.n_step_buffer.clear()

    def sample(self, batch_size):
        segment = self.sum_tree.total() / batch_size
        idxs = []
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx = self.sum_tree.get(s)
            idxs.append(idx)
            leaf_idx = idx + self.size  # Adjusting index to point to the leaf node
            priorities.append(self.sum_tree.tree[leaf_idx])

        priorities = torch.tensor(priorities, dtype=torch.float32, device=self.device).unsqueeze(1)
        sampling_probabilities = priorities / self.sum_tree.total()
        weights = (self.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        data = SimpleNamespace(
            observations=torch.from_numpy(self.observations[idxs]).to(self.device),
            next_observations=torch.from_numpy(self.next_observations[idxs]).to(self.device),
            actions=torch.from_numpy(self.actions[idxs]).to(self.device),
            rewards=torch.from_numpy(self.rewards[idxs]).to(self.device),
            dones=torch.from_numpy(self.dones[idxs]).to(self.device),
        )
        return data, idxs, weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.sum_tree.update(idx, priority)

    def update_beta(self, fraction):
        self.beta = (1.0 - self.beta_0) * fraction + self.beta_0


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        init.constant_(self.bias_mu, 0)
        init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon if self.training else self.weight_mu
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.training else self.bias_mu
        return F.linear(input, weight, bias)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n

        self.shared_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, n_atoms))
        self.advantage_stream = nn.Sequential(NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, self.n * n_atoms))

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_action(self, obs):
        q_values_distributions = self.get_distribution(obs)
        q_values = (torch.softmax(q_values_distributions, dim=2) * self.atoms).sum(2)
        return torch.argmax(q_values, 1)

    def get_distribution(self, obs):
        x = self.shared_layers(obs / 255.0)
        value = self.value_stream(x).view(-1, 1, self.n_atoms)
        advantages = self.advantage_stream(x).view(-1, self.n, self.n_atoms)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(args.buffer_size, device)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = q_network.get_action(torch.Tensor(obs).to(device))
        actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data, idxs, weights = rb.sample(args.batch_size)

                # Combine observations for a single network call
                combined_obs = torch.cat([data.observations, data.next_observations], dim=0)
                combined_dist = q_network.get_distribution(combined_obs)
                dist, next_dist = combined_dist.split(len(data.observations), dim=0)

                with torch.no_grad():
                    next_q_values = (torch.softmax(next_dist, dim=2) * q_network.atoms).sum(2)
                    next_actions = torch.argmax(next_q_values, 1)
                    target_next_dist = target_network.get_distribution(data.next_observations)
                    next_pmfs = torch.softmax(target_next_dist[torch.arange(len(data.next_observations)), next_actions], dim=1)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones.float())
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                old_pmfs = torch.softmax(dist[torch.arange(len(data.observations)), data.actions.flatten()], dim=1)

                expected_old_q = (old_pmfs.detach() * q_network.atoms).sum(-1)
                expected_target_q = (target_pmfs * target_network.atoms).sum(-1)
                td_error = expected_target_q - expected_old_q
                rb.update_priorities(idxs, td_error.abs().cpu().numpy())
                rb.update_beta(global_step / args.total_timesteps)

                loss = (weights * -(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log())).sum(-1).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", expected_old_q.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                q_network.reset_noise()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.rainbow_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "RAINBOW", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
