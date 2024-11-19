import argparse
import os

import numpy
import numpy as np
import torch
from transformers import enable_full_determinism

from src.searcher.BeamEnv import BeamSystem
from src.searcher.auto_turning_rl import DDPG

if __name__ == '__main__':
    all_find, all_iter = [], []
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, default='system2')
    parser.add_argument('--use_att', type=str, default='att')
    parser.add_argument('--d', type=float, default=0.1)
    parser.add_argument('--max_steps', type=float, default=10)
    parser.add_argument('--action_dim', type=int, default=30)
    args = parser.parse_args()
    use_att = args.use_att
    distance_threshold = args.d
    max_steps = args.max_steps
    data_set_name = args.system
    action_dim = args.action_dim
    for seed in [42, 43, 44]:
        actor_lr = 3e-5
        critic_lr = 3e-5
        hidden_dim = 500
        state_dim = 8
        if use_att == 'avg':
            action_bound = 1 / action_dim
        else:
            action_bound = 1
        sigma = 0.1
        tau = 0.005
        gamma = 0.98
        enable_full_determinism(seed=seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        system_model_path = './data/mock_system_model/{}_model.pth'.format(data_set_name)
        system_data_path = './data/system_data/{}_data.pkl'.format(data_set_name)
        agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, use_att)
        agent.actor.load_state_dict(torch.load('./data/searcher_model/{}_{}_{}_actor.pth'.format(data_set_name, use_att, seed)))
        agent.actor.eval()
        env = BeamSystem(input_size=action_dim, system_model_path=system_model_path, system_data_path=system_data_path, distance_threshold=distance_threshold)
        find = 0
        iter_time_list = []
        for i in range(500):
            state = env.reset()
            print('===================')
            print('start eval:[{}],init state:{}'.format(i, state))
            print('goal state:{}'.format(env.goal))
            step = 0
            while True:
                step += 1
                action = agent.take_action(state, False)
                next_state, reward, done = env.step(action)
                state = next_state
                print('step:{},state:{},mae:{}'.format(step, state[:4], -reward))
                if done:
                    find += 1
                    iter_time_list.append(step)
                    break
                if step >= max_steps:
                    iter_time_list.append(step)
                    break
        all_find.append(find)
        all_iter.append(numpy.mean(iter_time_list))
    print('for dataset:{}, distance:{}, att:{}, max_step:{}'.format(data_set_name, distance_threshold, use_att, max_steps))
    print("find:{}".format(str(numpy.mean(all_find))))
    print("iter_time_list:{}".format(numpy.mean(all_iter)))
