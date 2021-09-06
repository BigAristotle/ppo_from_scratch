import torch
import torch.nn as nn
from torch.distributions import Normal
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LOG_STD = 2
MIN_LOG_STD = -5

class Actor_Critic(nn.Module):
    def __init__(self, actor_in_dims, actor_hidden_sizes, actor_out_dims,
                 critic_in_dims, critic_hidden_sizes, critic_out_dim=1,
                 actor_activation=torch.tanh, critic_activation=torch.tanh):
        super(Actor_Critic, self).__init__()
        self.actor_mlp, self.critic_mlp = [], []
        self.actor_activation, self.critic_activation = actor_activation, critic_activation

        actor_layer_indims = actor_in_dims
        for i, actor_layer_outdims in enumerate(actor_hidden_sizes):
            actor_fc = nn.Linear(actor_layer_indims, actor_layer_outdims)
            self.actor_mlp.append(actor_fc)
            self.__setattr__("actor_layer_{}".format(i), actor_fc)
            actor_layer_indims = actor_layer_outdims

        self.actor_out_mean = nn.Linear(actor_layer_indims, actor_out_dims)
        self.__setattr__("actor_layer_out_mean", self.actor_out_mean)
        self.actor_out_log_std = nn.Linear(actor_layer_indims, actor_out_dims)
        self.__setattr__("actor_layer_out_log_std", self.actor_out_log_std)

        critic_layer_indims = critic_in_dims
        for i, critic_layer_outdims in enumerate(critic_hidden_sizes):
            critic_fc = nn.Linear(critic_layer_indims, critic_layer_outdims)
            self.critic_mlp.append(critic_fc)
            self.__setattr__("critic_layer_{}".format(i), critic_fc)
            critic_layer_indims = critic_layer_outdims

        self.critic_out_value = nn.Linear(critic_layer_indims, critic_out_dim)
        self.__setattr__("critic_layer_out_value", self.critic_out_value)

    def forward(self, curr_obs_tensor):
        actor_ = curr_obs_tensor
        for i, actor_layer in enumerate(self.actor_mlp):
            actor_ = self.actor_activation(actor_layer(actor_))
        mean = self.actor_out_mean(actor_)
        log_std = self.actor_out_log_std(actor_).clamp(MIN_LOG_STD, MAX_LOG_STD)

        critic_ = curr_obs_tensor
        for i, critic_layer in enumerate(self.critic_mlp):
            critic_ = self.actor_activation(critic_layer(critic_))
        value = self.critic_out_value(critic_)

        return mean, log_std.exp(), value

    def select_action(self, curr_obs_tensor):
        mean, std, value = self.forward(curr_obs_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        #log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action.squeeze(0).cpu().numpy(), log_prob.detach().cpu().numpy(), value.squeeze(0).detach().cpu().numpy()

    def compute_action(self, obs_idx_tensor, actions_idx_tensor):
        mean, std, value = self.forward(obs_idx_tensor)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions_idx_tensor).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)  # 1/2*（ln det(2pi*e*∑)
        return log_prob, entropy, value