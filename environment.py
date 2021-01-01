import gym
import numpy as np

# TODO: use logger with log level
DEBUG = True
# TODO: config file or process params
RENDER = False

# TODO: separate in files per classes

# A table of discrete buckets for each observation vector and action, which holds the expected reward fot taking
# such action coming from that observation combination.
class QTable:
    NUMBER_OF_BUCKETS = 20
    def __init__(self, environment):
        self.buckets = [self.NUMBER_OF_BUCKETS] * len(environment.observation_space.high)
        self.bucket_sizes = (environment.observation_space.high - environment.observation_space.low) / self.buckets
        # Initialize table with randoms for each bucket and action
         # Low and high depends on rewards, for mountain its -1 if fail, 0 if success
        self.table = np.random.uniform(low=-2, high=0, size=(self.buckets + [environment.action_space.n]))
        if DEBUG: self.describe_table()
    
    def describe_table(self):
        print("------ QTable ------")
        print("Bucket space description", self.buckets)
        print("Bucket sizes", self.bucket_sizes)
        print("Table info", self.table.shape)

class Environment:
    ITERATIONS_IN_EPISODE = 100
    def __init__(self, environmentName):
        self.env = gym.make(environmentName)
        self.qTable = QTable(self.env)
        if DEBUG : self.describe_environment()

    def describe_environment(self):
        print("------ Environment Description ------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)

    def run_episode(self):
        if DEBUG : print("------ Starting Episode ------")
        self.env.reset()
        for _ in range(self.ITERATIONS_IN_EPISODE):
            # 0 left 2 right 1 nothing
            action = 2
            if RENDER : self.env.render()
            new_state, reward, done, _ = self.env.step(self.env.action_space.sample())
            # print(new_state)
    
    def close(self):
        if DEBUG: print("------ Done ------")
        self.env.close()

env = Environment("MountainCar-v0")
env.run_episode()
env.close()