# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/iqn/#iqn_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
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
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--num-cosines", type=int, default=64,
        help="the number of cosines")
    parser.add_argument("--num-quantile-samples", type=int, default=32,
        help="the number of quantile samples")
    parser.add_argument("--num-tau-samples", type=int, default=64,
        help="the number of tau samples")
    parser.add_argument("--num-tau-prime-samples", type=int, default=64,
        help="the number of tau prime samples")
    args = parser.parse_args()
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_cosines, 3136),
            nn.ReLU(),
        )
        i_pi = np.pi * torch.arange(start=1, end=num_cosines + 1).reshape(1, 1, num_cosines)  # (1, 1, num_cosines)
        self.register_buffer("i_pi", i_pi)

    def forward(self, taus):
        # Compute cos(i * pi * tau)
        taus = torch.unsqueeze(taus, dim=-1)  # (batch_size, num_tau_samples, 1)
        cosines = torch.cos(taus * self.i_pi)  # (batch_size, num_tau_samples, num_cosines)

        # Compute embeddings of taus
        cosines = torch.flatten(cosines, end_dim=1)  # (batch_size * num_tau_samples, num_cosines)
        tau_embeddings = self.network(cosines)  # (batch_size * num_tau_samples, embedding_dim)
        return torch.unflatten(
            tau_embeddings, dim=0, sizes=(-1, taus.shape[1])
        )  # (batch_size, num_tau_samples, embedding_dim)


class QuantileNetwork(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, embeddings, tau_embeddings):
        # Compute the embeddings and taus
        embeddings = torch.unsqueeze(embeddings, dim=1)  # (batch_size, 1, embedding_dim)
        embeddings = embeddings * tau_embeddings  # (batch_size, M, embedding_dim)

        # Compute the quantile values
        embeddings = torch.flatten(embeddings, end_dim=1)  # (batch_size * M, embedding_dim)
        quantiles = self.network(embeddings)
        return torch.unflatten(quantiles, dim=0, sizes=(-1, tau_embeddings.shape[1]))  # (batch_size, M, num_actions)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs = RecordEpisodeStatistics(envs)
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork().to(device)
    target_q_network = QNetwork().to(device)
    cosine_network = CosineEmbeddingNetwork(args.num_cosines).to(device)
    target_cosine_network = CosineEmbeddingNetwork(args.num_cosines).to(device)
    quantile_network = QuantileNetwork(envs).to(device)
    target_quantile_network = QuantileNetwork(envs).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    target_cosine_network.load_state_dict(cosine_network.state_dict())
    target_quantile_network.load_state_dict(quantile_network.state_dict())

    optimizer = optim.Adam(
        list(q_network.parameters()) + list(cosine_network.parameters()) + list(quantile_network.parameters()),
        lr=args.learning_rate,
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            embeddings = q_network(torch.Tensor(obs).to(device))
            taus = torch.rand(envs.num_envs, args.num_quantile_samples, device=device)
            tau_embeddings = cosine_network(taus)
            quantiles = quantile_network(embeddings, tau_embeddings)  # (num_quantile_samples, num_actions)
            q_values = torch.mean(quantiles, dim=1)  # (num_actions,)
            actions = torch.argmax(q_values, dim=1, keepdim=True).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # Sample fractions and compute quantile values of current observations and actions at taus
                embeddings = q_network(data.observations)
                taus = torch.rand(args.batch_size, args.num_tau_samples, device=device)
                tau_embeddings = cosine_network(taus)
                quantiles = quantile_network(embeddings, tau_embeddings)

                # Compute quantile values at specified actions. The notation seems eavy notation,
                # but just select value of s_quantile (batch_size, num_tau_samples, num_quantiles) with
                # action_indexes (batch_size, num_quantile_samples).
                # Output shape is thus (batch_size, num_quantile_samples)
                action_index = data.actions.expand(-1, args.num_tau_samples)  # Expand to (batch_size, num_tau_samples)
                current_action_quantiles = quantiles.gather(dim=2, index=action_index.unsqueeze(-1)).squeeze(-1)

                # Compute Q values of next observations
                next_embeddings = target_q_network(data.next_observations)
                next_taus = torch.rand(args.batch_size, args.num_quantile_samples, device=device)
                next_tau_embeddings = target_cosine_network(next_taus)
                next_quantiles = target_quantile_network(next_embeddings, next_tau_embeddings)

                # Compute greedy actions
                next_q_values = torch.mean(next_quantiles, dim=1)  # (batch_size, num_actions)
                next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)  # (batch_size,)

                # Compute next quantiles
                tau_dashes = torch.rand(args.batch_size, args.num_tau_prime_samples, device=device)
                tau_dashes_embeddings = target_cosine_network(tau_dashes)
                next_quantiles = target_quantile_network(next_embeddings, tau_dashes_embeddings)

                # Compute quantile values at specified actions. The notation seems eavy notation,
                # but just select value of s_quantile (batch_size, num_tau_samples, num_quantiles)
                # with action_indexes (batch_size, num_quantile_samples).
                # Output shape is thus (batch_size, num_quantile_samples).
                next_action_index = next_actions.expand(-1, args.num_tau_prime_samples)
                next_action_quantiles = next_quantiles.gather(dim=2, index=next_action_index.unsqueeze(-1)).squeeze(-1)

                # Compute target quantile values (batch_size, num_tau_prime_samples)
                target_action_quantiles = data.rewards + args.gamma * next_action_quantiles * (1 - data.dones)

                # TD-error is the cross differnce between the target quantiles and the currents quantiles
                td_errors = target_action_quantiles.unsqueeze(-2).detach() - current_action_quantiles.unsqueeze(-1)

                # Compute quantile Huber loss
                huber_loss = torch.where(torch.abs(td_errors) <= 1.0, td_errors**2, (torch.abs(td_errors) - 0.5))
                quantile_huber_loss = torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * huber_loss
                batch_quantile_huber_loss = torch.sum(quantile_huber_loss, dim=1)
                loss = torch.mean(batch_quantile_huber_loss)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", current_action_quantiles.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for network, target_network in [
                    (q_network, target_q_network),
                    (cosine_network, target_cosine_network),
                    (quantile_network, target_quantile_network),
                ]:
                    for target_network_param, network_param in zip(target_network.parameters(), network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.iqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "IQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
