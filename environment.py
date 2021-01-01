import gym

ITERATIONS_IN_EPISODE = 100
DEBUG = True
RENDER = False

class QTable:
    def __init__(self, environment):
        self.discrete_os_size =  [20] + len(environment.observation_space.high)

class Environment:
    def __init__(self, environmentName):
        self.env = gym.make(environmentName)
        if DEBUG : self.describe_environment()

    def describe_environment(self):
        print("-------- Environment Description --------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)
        print("--------")

    def run_episode(self):
        if DEBUG : print("-------- Starting Episode --------")
        self.env.reset()
        for _ in range(ITERATIONS_IN_EPISODE):
            # 0 left 2 right 1 nothing
            action = 2
            if RENDER : self.env.render()
            new_state, reward, done, _ = self.env.step(self.env.action_space.sample())
            print(new_state)
    
    def close(self):
        self.env.close()

env = Environment("MountainCar-v0")
env.run_episode()
env.close()