
from copy import copy, deepcopy
import numpy as np

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AtariPreprocessing
import ale_py

import torch

from tqdm import tqdm
from tqdm.contrib import concurrent
from multiprocessing import shared_memory
from itertools import repeat

from mgymnasium.util import get_gymInfo
from mtorch.Rainbow.AtariModule import AtariFramePreprocesser, AtariNoisyQnetwork
from mtorch.Rainbow.Rainbow import RainbowAgent
import mtorch.Rainbow.util as Rainbow_util
from ReplayBuffer.DistributedBuffer import UniformDistributedReplayBuffer
from usefulParam.Param import makeConstant, makeMultiply
from mgymnasium.AutoFireBreakout import AutoFireBreakout

def wormup_worker(env: Env, action_kinds, afpp: AtariFramePreprocesser):

    done = False
    observations = list()
    state, _ = env.reset()
    state = np.expand_dims(state, axis=0)
    state = afpp.preprocessing(state)
    while not done:
        # 行動選択
        action = np.random.choice(range(4))
        # action = agent.get_action(torch.tensor(state).to(dtype=torch.float))

        # 実行
        next_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated or terminated
        next_state = np.expand_dims(next_state, axis=0)
        next_state = afpp.preprocessing(next_state)

        # エージェントを更新
        observations.append([state, action, reward, next_state, done])

        # 後処理
        state = next_state
    return observations

def main():
    args = {
        # env
        'env_name': "ALE/Breakout-v5", 
        # hyper param
        'episodes': 10000, 
        'gamma': 0.995, 
        'lr': 1e-4, 
        'tau': 1e-3, 
        'epsilon_init': 1.0, 
        'epsilon_dest': 1e-3, 
        'epsilon_dcayRate': 0.999, 
        'warmup_episode': 1000, 
        # qnet
        'frame_num': 4, 
        # replayBuf
        'capacity': 100000, 
        'batch_size': 16, 
        # device
        'device': 'cpu',
        # result
        'project': 'project', 
        'result': 'result'
    }

    device = torch.device(args['device'])
    env_name = args['env_name']
    episodes = args['episodes']
    gamma = makeConstant(args['gamma'], device)
    lr = makeConstant(args['lr'], device)
    tau = makeConstant(args['tau'], device)
    epsilon = makeMultiply(args['epsilon_init'], args['epsilon_dcayRate'], args['epsilon_dest'], args['epsilon_init'], device)
    warmup_episode = args['warmup_episode']
    frame_num = args['frame_num']
    capacity = args['capacity']
    batch_size = args['batch_size']
    project = args['project']
    result = args['result']

    env = AutoFireBreakout()
    state_size, action_size, action_kinds, clearScoreThreshold = get_gymInfo(env_name)

    qnet = AtariNoisyQnetwork(state_size, frame_num, action_kinds, dueling=True, do_noiseReset=True)
    target_qnet = AtariNoisyQnetwork(state_size, frame_num, action_kinds, dueling=True, do_noiseReset=False)

    replayBuf = UniformDistributedReplayBuffer(
        capacity, 
        state_size, action_size, 
        state_type=torch.float, action_type=torch.int, 
        device=device
    )

    agent = RainbowAgent(
        gamma, lr, tau, epsilon, 
        state_size, action_size, action_kinds, 
        qnet, target_qnet, 
        'MSELoss', 'Adam', 1, 
        replayBuf=replayBuf, batch_size=batch_size, 
    )

    afpp = AtariFramePreprocesser(frame_num, (84, 84), device)

    # wormup
    # wormup_worker(env, action_kinds, afpp)
    print('warmup')
    episode_observations = concurrent.process_map(wormup_worker, repeat(env, warmup_episode), repeat(action_kinds, warmup_episode), repeat(afpp, warmup_episode), max_workers=10)
    print('add warmup observation to buffer')
    for observations in tqdm(episode_observations, ncols=80):
        for observation in observations:
            agent.add_buffer(*observation)
    env = AutoFireBreakout(human_render=True)

    # loop
    reward_history = list()
    for episode in tqdm(range(episodes), ncols=80):
        state, _ = env.reset()
        state = afpp.preprocessing(np.expand_dims(state, axis=0))
        done = False
        total_reward = 0.0
        action_history = [0 for _ in range(action_kinds)]

        while not done:
            # 行動選択
            # action = np.random.choice(range(4))
            action = agent.get_action(torch.tensor(state).to(dtype=torch.float))
            action = np.array(action).item()
            action_history[action] += 1

            # 実行
            next_state, reward, truncated, terminated, _ = env.step(action)
            done = truncated or terminated
            next_state = np.expand_dims(next_state, axis=0)
            next_state = afpp.preprocessing(next_state)

            # エージェントを更新
            state = np.array(state)
            action = np.array(action)
            reward = np.array(reward)
            next_state = np.array(next_state)
            done = np.array(done)
            agent.update(state, action, reward, next_state, done)

            # 後処理
            state = next_state
            total_reward += reward
        
        reward_history.append(total_reward)
        tqdm.write(f'{episode}: reward={total_reward}')
        tqdm.write(f'action history: {action_history}')
        tqdm.write(f'epsilon: {agent._epsilon.value}')

        agent.param_step()