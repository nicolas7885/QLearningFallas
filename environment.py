import gym
import numpy as np
import yaml
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
DEBUG = config["run"]["debug"]
RENDER = config["run"]["render"]

# TODO: separate in files per classes

# A table of discrete buckets for each observation vector and action, which holds the expected reward fot taking
# such action coming from that observation combination.
# Discrete process is necessary to turn continous environment into a manageable one
class QTable:
    NUMBER_OF_BUCKETS = config["qTable"]["numberOfBuckets"]
    # How much of new info overrides old. priorize old -> `0 <= x <= 1` -> priorize new
    LEARNING_RATE = config["qTable"]["learningRate"]
    # How much we look to the future. immediate reward -> `0 <= x < 1` -> future rewards.
    DISCOUNT_FACTOR = config["qTable"]["discountFactor"]

    def __init__(self, environment):
        self.low = environment.observation_space.low
        self.high = environment.observation_space.high
        self.buckets = [self.NUMBER_OF_BUCKETS] * len(self.high)
        self.bucket_sizes = (self.high - self.low) / self.buckets
        # Initialize table with randoms for each bucket and action
         # Low and high depends on rewards, for mountain its -1 if fail, 0 if success
        self.table = np.random.uniform(low=-2, high=0, size=(self.buckets + [environment.action_space.n]))
        if DEBUG: self.describe_table()
    
    def describe_table(self):
        print("------ QTable ------")
        print("Bucket space description", self.buckets)
        print("Bucket sizes", self.bucket_sizes)
        print("Table info", self.table.shape)

    def get_discrete(self, state):
        discrete = (state - self.low) / self.bucket_sizes
        return tuple(discrete.astype(np.int))

    def get_best_action(self, current_state):
        return np.argmax(self.table[current_state])

    def update_q(self, current_state, action, new_state, reward):
        max_future_q = np.max(self.table[new_state])
        current_q_index = current_state + (action, )
        current_q = self.table[current_q_index]
        new_q = 1 - self.LEARNING_RATE * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * max_future_q)
        self.table[current_q_index] = new_q

    def mark_state_as_win(self, state, action):
        self.table[state + (action, )] = 0


class Environment:
    ITERATIONS_IN_EPISODE = config["environment"]["iterations"]
    def __init__(self, environmentName):
        self.env = gym.make(environmentName)
        self.qTable = QTable(self.env)
        self.render = False

    def describe_environment(self):
        print("------ Environment Description ------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)
        print("Goal", self.env.goal_position)

    def run_episode(self):
        current_discrete_state = self.qTable.get_discrete(self.env.reset())
        for iteration in range(self.ITERATIONS_IN_EPISODE):
            if self.render : self.env.render()
            action = self.qTable.get_best_action(current_discrete_state)
            new_state, reward, done, _ = self.env.step(action)
            new_discrete_state = self.qTable.get_discrete(new_state)
            if not done: 
                self.qTable.update_q(current_discrete_state, action, new_discrete_state, reward)
            elif new_state[0] >= self.env.goal_position :
                if DEBUG: print("Won on iteration", iteration)
                self.qTable.mark_state_as_win(current_discrete_state, action)
                break
    
    def close(self):
        if DEBUG: print("------ Done ------")
        self.env.close()

    def set_render(self, shouldRender):
        self.render = shouldRender

env = Environment(config["environment"]["name"])
if DEBUG : env.describe_environment()
for episode in range(config["run"]["episodes"]):
    if DEBUG : print(f"------ Starting Episode {episode}------")
    if config["run"]["render"]:
        env.set_render(episode % config["run"]["renderEvery"] == 0)
    env.run_episode()
env.close()