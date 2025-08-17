
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
        'episodes': 100, 
        'gamma': 0.995, 
        'lr': 1e-4, 
        'tau': 1e-3, 
        # qnet
        'frame_num': 4, 
        # replayBuf
        'capacity': 1000000, 
        'batch_size': 128, 
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
    frame_num = args['frame_num']
    capacity = args['capacity']
    batch_size = args['batch_size']
    project = args['project']
    result = args['result']

    env = AutoFireBreakout()
    state_size, action_kinds, action_size, clearScoreThreshold = get_gymInfo(env_name)

    qnet = AtariNoisyQnetwork(state_size, frame_num, action_kinds, dueling=True)
    target_qnet = deepcopy(qnet)
    target_qnet.dueling = False

    replayBuf = UniformDistributedReplayBuffer(
        capacity, 
        state_size, action_size, 
        state_type=torch.int8, action_type=torch.int8, 
        device=device
    )

    agent = RainbowAgent(
        gamma, lr, tau, 
        state_size, action_size, action_kinds, 
        qnet, target_qnet, 
        'MSELoss', 'Adam', 1, 
        replayBuf=replayBuf, batch_size=batch_size, 
    )

    afpp = AtariFramePreprocesser(4, (84, 84), device)

    # wormup
    # wormup_worker(env, action_kinds, afpp)
    episode_observations = concurrent.process_map(wormup_worker, repeat(env, 500), repeat(action_kinds, 500), repeat(afpp, 500), max_workers=10)

    # loop
    reward_history = list()
    for episode in range(episodes):
        state, _ = env.reset()
        state = afpp.preprocessing(np.expand_dims(state, axis=0))
        done = False
        total_reward = 0.0

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

            # 後処理
            state = next_state