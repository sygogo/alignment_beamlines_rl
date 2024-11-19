import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from src.searcher.BeamEnv import BeamSystem
from train_mock import SystemModel
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, use_att):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.use_att = use_att
        if use_att == 'att':
            self.att = torch.nn.Linear(hidden_dim, action_dim)
            nn.init.xavier_uniform_(self.att.weight)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        if self.use_att == 'att':
            att = self.softmax(self.att(x))
            # print(att.detach().cpu().numpy().tolist())
            return torch.tanh(self.fc3(x)) * att * self.action_bound
        else:
            return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc2(F.relu(self.fc1(cat))))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, use_att):
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, use_att).cuda()
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).cuda()
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, use_att).cuda()
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).cuda()
        # 初始化目标价值网络并使其参数和价值网络一样
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并使其参数和策略网络一样
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound

    def take_action(self, state, with_noise=True):
        state = torch.tensor(state, dtype=torch.float).cuda()
        action = self.actor(state.unsqueeze(0)).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        if with_noise:
            action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).cuda()
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).cuda()
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).cuda()
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).cuda()
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).cuda()

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # MSE损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 策略网络就是为了使Q值最大化
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


class Trajectory:
    ''' 用来记录一条完整轨迹 '''

    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, dis_threshold=0.15, her_ratio=0.8):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal = traj.states[step_goal][:4]  # 使用HER算法的future方案设置目标
                # dis = mean_absolute_error(goal, next_state[:4])
                dis1 = mean_absolute_error(goal[:2], next_state[:4][:2])
                dis2 = mean_absolute_error(goal[2:], next_state[:4][2:])
                dis = dis1 + 2 * dis2
                reward = -dis
                if dis < dis_threshold:
                    done = True
                else:
                    done = False
                state = np.hstack((state[:4], goal))
                next_state = np.hstack((next_state[:4], goal))

            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        return batch
