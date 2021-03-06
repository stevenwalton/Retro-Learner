import gym
from gym import wrappers
import retro

class RandomAgent(gym.Wrapper):
    def __init__(self, 
                 game,
                 render,
                 state,
                 scenario,
                 save,
                 max_episodes=1000,
                 ):
        self.game = game
        self.max_episodes = max_episodes
        self.state = state
        self.scenario = scenario
        self.max_reward = -1 
        self.render = render
        self.save=save

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def run(self):
        self.env = retro.make(game=self.game,
                              state=self.state,
                              scenario=self.scenario)
        self.action_space = self.env.action_space
        reward = 0
        total_reward = 0
        done = False
        for i in range(self.max_episodes):
            ob = self.env.reset()
            while True:
                action = self.act(ob, reward, done)
                if (self.render):
                    self.env.render()
                ob, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    if total_reward > self.max_reward:
                        print("Current Max Reward:",total_reward)
                        self.max_reward = total_reward
                        if (self.save is not None):
                            print("Saving is not currently allowed")
                    total_reward = 0
                    reward = 0
                    break
        self.env.close()
