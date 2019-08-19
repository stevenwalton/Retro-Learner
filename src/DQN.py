'''
Our Deep Q-Learning algo
'''
import time
import retro
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torchvision.transforms as T

import Node
import utils

class DQN_Net(nn.Module):
    def __init__(self):
        super(DQN_Net, self)__init__()
        self.fc1 = nn.Linear(n_states, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30, n_actions)
        self.out.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.out(x)

class DQN:
    def __init__(self,
                 env,
                 max_episode_steps,
                 render,
                 discount,
                 gamma,
                 ):
        self.node_count = 1
        self.root = Node.Node()
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.render = render
        self.discount = discount
        self.gamma = gamma
        
    def run(self):
        acts = self.select_actions()
        steps, total_rewards = utils.rollout(acts)
        executed_acts = acts[:steps]
        self.node_count += utils.update_tree(executed_acts, total_rewards)
        return executed_acts, total_rewards

class DQNAgent():
    def __init__(self,
                 game,
                 state,
                 scenario,
                 discount,
                 gamma,
                 save,
                 fs_skip,
                 render,
                 timestep_limit,
                 max_episode_steps=4500,
                 time=False,
                 ):
        self.game = game
        self.max_episode_steps = max_episode_steps
        self.timestep_limit = int(timestep_limit)
        self.state = state
        self.discount = discount
        self.gamma = gamma
        self.scenario = scenario
        self.save = save
        self.render = render
        self.time = time
        if save is not None and ".bk2" not in save.save[-4:]:
            save.save += ".bk2"

        self.timesteps = 0
        self.best_reward = float('-inf')

        self.env = retro.make(game=game,
                              state=state,
                              use_restricted_actions=retro.Actions.DISCRETE,
                              scenario=scenario)
        if fs_skip > 0:
            self.env = FrameSkip.Frameskip(self.env, skip=fs_skip)
        self.env = TimeLimit.TimeLimit(self.env,
                                       max_episode_steps=max_episode_steps)

    def start(self):
        dqn = DQN(pass)
        if self.time:
            startTime = time.time()
        while True:
            acts, reward = dqn.run()
            self.timesteps += len(acts)
            if reward > self.best_reward:
                print(f"New best reward {reward} from {self.best_reward}")
                if self.time:
                    print(f"Elapsed time {time.time() - startTime}")
                self.best_reward = reward
                if self.save is not None:
                    print("Saving {self.save}")
                    self.env.unwrapped.record_movie(self.save)
                    self.env.reset()
                    for act in acts:
                        self.env.step(act)
                    self.env.unwrapped.stop_record()
            if self.timesteps > self.timestep_limit:
                print("Timed out")
                break
