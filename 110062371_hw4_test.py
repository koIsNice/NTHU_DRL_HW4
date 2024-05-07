import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
#-----------------------------------------------------------------

class Q_Network(nn.Module):
    def __init__(self, num_actions, input_dim) -> None:
        super(Q_Network, self).__init__()

        self.linear1 = nn.Linear(input_dim + num_actions, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)

        

    def forward(self, observation, action):
        x = nn.Flatten()(observation)
        x = torch.cat([x, action], 1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

class Policy_Net(nn.Module):
    def __init__(self, num_actions, input_dim) -> None:
        super(Policy_Net, self).__init__()

        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, 1024)

        self.mu = nn.Linear(1024, num_actions)
        self.log_std = nn.Linear(1024, num_actions)

    def forward(self, observation):

        x = nn.Flatten()(observation)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

class SAC:
    def __init__(self, n_actions, reward_scale, entropy_coe, dicounted_factor, soft_update_coe, input_dim, device:torch.device, q_learning_rate=3e-4, policy_learning_rate=3e-4) -> None:
        self.n_actions = n_actions
        self.device = device
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.reward_scale = reward_scale
        self.entropy_coe = entropy_coe
        self.gamma = dicounted_factor
        self.tau = soft_update_coe
        self.input_dim = input_dim

        # self.target_entropy = -self.n_actions
        # self.log_entorpy_alpha = torch.tensor(0, dtype=torch.float32, requires_grad=True, device=self.device)
        # self.entropy_alpha = self.log_entorpy_alpha.exp().item()


        self.q_net = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net2 = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net_target = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net2_target = Q_Network(num_actions=self.n_actions, input_dim=input_dim).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())

        self.policy_net = Policy_Net(num_actions=self.n_actions, input_dim=input_dim).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.q_learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q_net2.parameters(), lr=self.q_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)
        #self.alpha_optimizer = torch.optim.Adam([self.log_entorpy_alpha], lr=1e-3)
        self.loss_function = nn.MSELoss()
        self.loss_function2 = nn.MSELoss()

    def load_nets(self, dict_q1, dict_q2, dict_policy):
        self.q_net.load_state_dict(dict_q1)
        self.q_net2.load_state_dict(dict_q2)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())
        self.policy_net.load_state_dict(dict_policy)

    def save_nets(self):
        torch.save(dict(q1=self.q_net.state_dict(), q2=self.q_net2.state_dict(), policy=self.policy_net.state_dict()), '110062371_hw4_data')

    def get_action_and_log_prob(self, observation):
        '''
        observation-> (batch_size, ...)
        action0 -> -1~1
        action1 -> 0~1
        action2 -> 0~1
        '''
        mean, log_std = self.policy_net(observation)
        std = log_std.exp()

        z = Normal(0, 1).sample(mean.shape).to(self.device)
        #print(z.shape, std.shape)

        unscaled_action = mean + std*z
        action = torch.tanh(unscaled_action)

        '''
        1st action : a = tanh(u) ; da/du = 1 - tanh(u) ^2
        2~3 actions: a = (tanh(u) + 1) / 2 ; da/du = (1 - tanh(u) ^2) / 2 => log(da/du) = log(1 - tanh(u) ^2) - log(2)

        we obtain the log prob. of a by applying Jocobian
        ''' 

        log_prob = Normal(mean, std).log_prob(unscaled_action) - torch.log(1 - action ** 2 + 1e-6)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob

    def update_nets(self, action, current_observation, next_observation, reward, done):
        #-----------------------------------------------------------------------------------
        # update Q nets
        
        q1 = self.q_net(current_observation, action)
        q2 = self.q_net2(current_observation, action)
        
        predicted_action, log_prob = self.get_action_and_log_prob(current_observation)
        predicted_next_action, next_log_prob = self.get_action_and_log_prob(next_observation)
        q1_target = self.q_net_target(next_observation, predicted_next_action)
        q2_target = self.q_net2_target(next_observation, predicted_next_action)

        # reward scaling
        #reward = self.reward_scale * (reward - reward.mean().item()) / (reward.std().item() + 1e-6)
        y = reward + (1. - done.float()) * self.gamma * (torch.min(q1_target, q2_target) - self.entropy_coe * next_log_prob)
        
        loss1 = self.loss_function(q1, y.detach())
        self.q_optimizer.zero_grad()
        loss1.backward()
        self.q_optimizer.step()

        loss2 = self.loss_function2(q2, y.detach())
        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        #-----------------------------------------------------------------------------------
        # update policy
        q1_policy = self.q_net(current_observation, predicted_action)
        q2_policy = self.q_net2(current_observation, predicted_action)
        selected_q = torch.min(q1_policy, q2_policy)

        policy_loss = (self.entropy_coe * log_prob - selected_q).mean()
        #print(self.entropy_coe * log_prob, selected_q)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #alpha_loss = -(self.log_entorpy_alpha * (log_prob + self.target_entropy).detach())

        # self.alpha_optimizer.zero_grad()
        # alpha_loss = alpha_loss.mean()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()
        #self.entropy_alpha = self.log_entorpy_alpha.exp().item()
        #-----------------------------------------------------------------------------------
        # soft update
        q1_dict = self.q_net.state_dict()
        q2_dict = self.q_net2.state_dict()
        q1_target_dict = self.q_net_target.state_dict()
        q2_target_dict = self.q_net2_target.state_dict()
        for key in q1_dict:
            q1_target_dict[key] = q1_target_dict[key] * (1. - self.tau) + q1_dict[key] * self.tau


        for key in q2_dict:
            q2_target_dict[key] = q2_target_dict[key] * (1. - self.tau) + q2_dict[key] * self.tau

        self.q_net_target.load_state_dict(q1_target_dict)
        self.q_net2_target.load_state_dict(q2_target_dict)

        #print(y, reward)

        return y, torch.min(q1, q2), loss1.item(), loss2.item(), policy_loss.item()
    
    @torch.no_grad()
    def get_q_values(self, action, current_observation, next_observation, reward, done):
        q1 = self.q_net(current_observation, action)
        q2 = self.q_net2(current_observation, action)
        q = torch.min(q1, q2)

        predicted_next_action, next_log_prob = self.get_action_and_log_prob(next_observation)
        q1_target = self.q_net_target(next_observation, predicted_next_action)
        q2_target = self.q_net2_target(next_observation, predicted_next_action)

        y = reward + (1. - done.float()) * self.gamma * (torch.min(q1_target, q2_target) - self.entropy_coe * next_log_prob)
        return y, q

class Agent:
    def __init__(self) -> None:

        skip = 4
        n_actions = 22
        len_of_obs_vec = 339
        reward_scale = 20.
        discounted_factor = .99
        entropy_coe = 1. / reward_scale
        soft_update_tau = 0.005
        batch_size = 256
        discounted_factor = .99
        device = torch.device("cpu")

        self.input_channels = skip
        self.n_actions = n_actions
        self.gamma = discounted_factor
        self.early_exploration = 0
        self.batch_size = batch_size
        self.input_dim = len_of_obs_vec

        self.current_obs = None
        self.counter = 0

        self.device = device
        self.sac = SAC(n_actions=n_actions, 
                       reward_scale=reward_scale, 
                       entropy_coe=entropy_coe, 
                       dicounted_factor=discounted_factor,
                       input_dim=self.input_dim, 
                       soft_update_coe=soft_update_tau,
                       device=device,
                       q_learning_rate=1e-4,
                       policy_learning_rate=3e-5)
        
        nets_history = torch.load('110062371_hw4_data', map_location=torch.device('cpu'))
        q1_dict=nets_history['q1']
        q2_dict=nets_history['q2']
        policy_dict=nets_history['policy']

        self.sac.load_nets(dict_q1= q1_dict, dict_q2=q2_dict, dict_policy=policy_dict)

    def unpack_dict_obs(self, observation:dict):
        res = []
        if not isinstance(observation, dict):
            if not (isinstance(observation, np.ndarray) or isinstance(observation, list)):
                res.append(observation)
            else:
                for element in observation:
                    res = res + self.unpack_dict_obs(element)
            return res
        
        for key in observation:
            res = res + self.unpack_dict_obs(observation[key])

        return res

    @torch.no_grad()
    def act(self, observation):

        if self.counter == 0:
            observation =self.unpack_dict_obs(observation)
            self.current_obs = observation

        observation = torch.tensor(observation, device=self.device, dtype=torch.float).unsqueeze(0)

        action, _ = self.sac.get_action_and_log_prob(observation=observation)

        action = action.squeeze().detach().cpu().__array__()
        scaled_action = action.copy()
        scaled_action[1] = (scaled_action[1] + 1.) / 2.
        scaled_action[2] = (scaled_action[2] + 1.) / 2.
        scaled_action = np.clip(scaled_action, 0., 1.)

        return scaled_action
#-----------------------------------------------------------------

#env = L2M2019Env(difficulty=2, visualize=False)
# env = wrapper.Wrapper_For_SkipFrame(env, skip)
# env = wrapper.Vector_Observation(env)
#obs = env.reset()

#-----------------------------------------------------------------
# training

# cumulative_reward = 0.

# agent = Agent()


# obs = env.reset()

# ep_len = 0
# while True:
#     action = agent.act(obs)

#     next_obs, reward, done, info = env.step(action)

#     cumulative_reward += reward
#     ep_len += 1

#     obs = next_obs

#     if done:
#         break

