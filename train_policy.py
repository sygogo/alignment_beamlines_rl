import argparse
import os

import torch
import random
import numpy as np
from torch.utils.hipify.hipify_python import str2bool
from tqdm import tqdm
from transformers import enable_full_determinism

from src.searcher.BeamEnv import BeamSystem
from src.searcher.auto_turning_rl import ReplayBuffer_Trajectory, DDPG, Trajectory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_att', type=str, default='avg', choices=['att', 'none_att', 'avg'])
    parser.add_argument('--local_rank', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local_rank
    data_set_name = 'system1'
    use_att = args.use_att
    actor_lr = 3e-4
    critic_lr = 3e-4
    hidden_dim = 500
    state_dim = 8
    action_dim = 12
    if use_att == 'avg':
        action_bound = 1 / action_dim
    else:
        action_bound = 1
    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    num_episodes = 500
    n_train = 20
    batch_size = 256
    minimal_episodes = 10
    buffer_size = 10000
    enable_full_determinism(seed=args.seed)
    system_model_path = './data/mock_system_model/{}_model.pth'.format(data_set_name)
    system_data_path = './data/system_data/{}_data.pkl'.format(data_set_name)
    env = BeamSystem(input_size=action_dim, system_model_path=system_model_path, system_data_path=system_data_path, distance_threshold=0.1)
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, use_att)
    best_reward = -1e9
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes), desc='Iteration %d' % i, ncols=200) as pbar:
            for i_episode in range(int(num_episodes)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        transition_dict = replay_buffer.sample(batch_size, True, dis_threshold=env.distance_threshold, her_ratio=0.8)
                        agent.update(transition_dict)
                if (i_episode + 1) % 50 == 0:
                    mean_reward = np.mean(return_list[-50:])
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 50 * i + i_episode + 1),
                        'return':
                            '%.3f' % mean_reward,
                        'best-reward':
                            '%.3f' % best_reward,
                    })
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        torch.save(agent.actor.state_dict(), './data/searcher_model/{}_{}_{}_actor.pth'.format(data_set_name, use_att, args.seed))
                        torch.save(agent.target_actor.state_dict(), './data/searcher_model/{}_{}_{}_target_actor.pth'.format(data_set_name, use_att, args.seed))
                        torch.save(agent.critic.state_dict(), './data/searcher_model/{}_{}_{}_critic.pth'.format(data_set_name, use_att, args.seed))
                        torch.save(agent.target_critic.state_dict(), './data/searcher_model/{}_{}_{}_target_critic.pth'.format(data_set_name, use_att, args.seed))
                pbar.update(1)
