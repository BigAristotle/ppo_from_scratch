from utils.cmd_utils import common_arg_parser
from utils.buffer import ReplayBuffer
from networks.actor_critic import Actor_Critic
from policies.ppo import PPO
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    args = common_arg_parser()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env_name = 'BipedalWalker-v2'
    env = gym.make(env_name)
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.shape[0]

    random_config = {
        'random_seed': 1
    }

    env_config = {
        'name': env_name,
        'obs_dims': obs_dims,
        'act_dims': act_dims
    }

    buffer_config = {
        'buffer_size': 1024,
        'batch_size': 64
    }

    actor_config = {
        'hidden_sizes': [128, 128],
        'activation': torch.tanh
    }

    critic_config = {
        'hidden_sizes': [256, 256],
        'activation': torch.tanh,
        'out_dim': 1
    }

    ppo_config = {
        'num_episodes': 2000,
        'max_length_per_episode': 1000,
        'K_epochs': 20,
        'clip_epsilon': 0.2,
        'gamma': 0.99,
        'gaelambda': 0.95,
        'c1_VF': 0.5,
        'c2_entropy': 0.01,
        'lr': 3e-4,
    }

    training_output_config = {
        'record_interval': 10,
        'solved_reward': 200
    }

    if random_config['random_seed']:
        random_seed = random_config['random_seed']
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    pooling = ReplayBuffer(buffer_config['buffer_size'], env_config['obs_dims'],
                           env_config['act_dims'], buffer_config['batch_size'])

    actorcritic = Actor_Critic(env_config['obs_dims'], actor_config['hidden_sizes'], env_config['act_dims'],
                               env_config['obs_dims'], critic_config['hidden_sizes'],
                               critic_config['out_dim'], actor_config['activation'], critic_config['activation'])
    actorcritic.to(device, dtype=torch.float32)

    ppo = PPO(env, actorcritic, pooling=pooling, num_episodes=ppo_config['num_episodes'],
              max_steps_per_episode=ppo_config['max_length_per_episode'],
              clip_epsilon=ppo_config['clip_epsilon'], gamma=ppo_config['gamma'], gae_lambda=ppo_config['gaelambda'],
              c1_VF=ppo_config['c1_VF'], c2_entropy=ppo_config['c2_entropy'], lr=ppo_config['lr'],
              K_epochs=ppo_config['K_epochs'])

    record_episodes, rewards_whole_process = ppo.train(training_output_config['record_interval'],
                                                       training_output_config['solved_reward'])

    plt.plot(np.asarray(record_episodes), np.asarray(rewards_whole_process))
    plt.show()
