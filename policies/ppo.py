import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PPO(object):
    def __init__(self, env, actor_critic, pooling, num_episodes, max_steps_per_episode,
                 clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95, c1_VF=0.5, c2_entropy=0.01, lr=3e-4, K_epochs=4):
        self.env, self.actor_critic, self.pooling = env, actor_critic, pooling
        self.num_episodes, self.max_steps_per_episode = num_episodes, max_steps_per_episode
        self.clip_epsilon, self.gamma, self.gae_lambda = clip_epsilon, gamma, gae_lambda
        self.c1_VF, self.c2_entropy, self.lr, self.K_epochs = c1_VF, c2_entropy, lr, K_epochs

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        self.mseloss = nn.MSELoss()

    def train(self, record_interval, reward_solved):
        rewards_whole_process = []
        record_timesteps = []
        total_timesteps = 0
        running_rewards, interval_length = 0., 0.
        print('Training Process Begins!')

        for n_episode in range(self.num_episodes):
            reward_per_episode = 0.
            curr_obs = self.env.reset()
            for steps_episode in range(self.max_steps_per_episode):
                action, old_log_Prob, value = self.actor_critic.select_action(torch.tensor(curr_obs,dtype=torch.float32).unsqueeze(0).to(device))
                next_obs, reward, done, _ = self.env.step(action)
                running_rewards += reward
                reward_per_episode += reward
                total_timesteps += 1

                self.pooling.store(curr_obs, action, reward, done, old_log_Prob, value)
                curr_obs = next_obs

                if self.pooling.is_full:
                    returns = self.compute_gae(next_obs)
                    self.optimizing_process(returns)
                    self.pooling.clear()

                if done:
                    break

            interval_length += (steps_episode+1)

            rewards_whole_process.append(reward_per_episode)
            record_timesteps.append(total_timesteps)

            if running_rewards > record_interval*reward_solved:
                average_length = interval_length / record_interval
                average_rewards = running_rewards / record_interval
                print('Episode {} \t\t Avg length: {} \t\t Avg reward: {}'.format(n_episode+1, average_length, average_rewards))
                print('####### Solved #######')
                running_rewards = 0.
                interval_length = 0.
                break

            if (n_episode+1) % record_interval == 0:
                average_length = interval_length / record_interval
                average_rewards = running_rewards / record_interval
                print('Episode {} \t\t Avg length: {} \t\t Avg reward: {}'.format(n_episode+1, average_length, average_rewards))
                running_rewards = 0.
                interval_length = 0.

        print('Training Process Stops!')
        return record_timesteps, rewards_whole_process

    def compute_gae(self, next_obs):
        observations, actions = self.pooling.observations, self.pooling.actions
        rewards, dones = self.pooling.rewards, self.pooling.dones
        values, old_log_probs = self.pooling.values, self.pooling.old_logProbs

        _, _, next_value = self.actor_critic.select_action(torch.tensor(next_obs, dtype=torch.float32).reshape(1, -1).to(device))

        gae_returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        lastgaelam = 0.
        for t in reversed(range(rewards.shape[0])):
            nextnonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * nextnonterminal * next_value - values[t]
            next_value = values[t]

            lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            advs[t] = lastgaelam

        gae_returns = advs + values

        return torch.from_numpy(gae_returns).to(device)

    def optimizing_process(self, returns):
        observations_np, actions_np = self.pooling.observations, self.pooling.actions
        rewards_np, dones_np = self.pooling.rewards, self.pooling.dones
        values_np, old_log_probs_np = self.pooling.values, self.pooling.old_logProbs

        observations = torch.from_numpy(observations_np).to(device)
        actions = torch.from_numpy(actions_np).to(device)
        rewards = torch.from_numpy(rewards_np).to(device)
        dones = torch.from_numpy(dones_np).to(device)
        values = torch.from_numpy(values_np).to(device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(device)

        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        for _ in range(self.K_epochs):
            for idx in self.pooling.sample():
                new_log_probs, actor_entropy, new_values = self.actor_critic.compute_action(observations[idx], actions[idx])

                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                pg_losses1 = ratios * advs[idx]
                pg_losses2 = torch.clamp(ratios, 1.-self.clip_epsilon, 1.+self.clip_epsilon) * advs[idx]

                pg_losses = torch.min(pg_losses1, pg_losses2).mean()
                vf_losses = self.c1_VF * self.mseloss(new_values, returns[idx].detach())
                entropy_bonus = (self.c2_entropy * actor_entropy).mean()

                surrogate_loss = -(pg_losses - vf_losses + entropy_bonus)
                self.optimizer.zero_grad()
                surrogate_loss.backward()
                self.optimizer.step()