import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, maxlen, obs_dims, act_dims, batch_size):
        self.batch_size = batch_size
        self.capacity = maxlen

        self.observations = np.zeros((self.capacity, obs_dims),  dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dims),  dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1),  dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1),  dtype=np.float32)
        self.old_logProbs = np.zeros((self.capacity, 1),  dtype=np.float32)
        self.values = np.zeros((self.capacity, 1),  dtype=np.float32)

        self.buffer_size = 0
        self.ptr = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, curr_obs, curr_act, curr_r, curr_done, old_logP, curr_value):
        self.observations[self.ptr] = curr_obs
        self.actions[self.ptr] = curr_act
        self.rewards[self.ptr] = curr_r
        self.dones[self.ptr] = curr_done
        self.old_logProbs[self.ptr] = old_logP
        self.values[self.ptr] = curr_value

        self.buffer_size = min(self.buffer_size + 1, self.capacity)
        # self.ptr = (self.ptr + 1) % self.capacity
        self.ptr += 1

    def sample(self):
        assert self.buffer_size % self.batch_size == 0, 'batch_size ({}) should divide buffer_size ({}) evenly'.format(
            self.batch_size, self.buffer_size)
        return BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.batch_size, True)
    # def sample(self):
    #     idx = np.random.randint(0, self.buffer_size, self.batch_size)
    #
    #     batch_cur_obs = self.cur_obs[idx]
    #     batch_actions = self.actions[idx]
    #     batch_rewards = self.rewards[idx]
    #     batch_next_obs = self.next_obs[idx]
    #     batch_done = self.done[idx]
    #
    #     device = self.device
    #
    #     return {0: torch.FloatTensor(batch_cur_obs).to(device),
    #             1: torch.FloatTensor(batch_actions).to(device),
    #             2: torch.FloatTensor(batch_rewards).to(device),
    #             3: torch.FloatTensor(batch_next_obs).to(device),
    #             4: torch.FloatTensor(batch_done).to(device)}

    def clear(self):
        self.observations *= 0
        self.actions *= 0
        self.rewards *= 0
        self.dones *= 0
        self.old_logProbs *= 0
        self.values *= 0

        self.ptr = 0

    @property
    def is_full(self):
        return self.ptr == self.capacity
