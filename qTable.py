import numpy as np
import yaml
config = yaml.safe_load(open("config.yaml"))["qTable"]
#TODO avoid reading env variables directle, pass them instead, qtable should not know how env is built

# A table of discrete buckets for each observation vector and action, which holds the expected reward fot taking
# such action coming from that observation combination.
# Discrete process is necessary to turn continous environment into a manageable one
class QTable:
    NUMBER_OF_BUCKETS = config["numberOfBuckets"]
    # How much of new info overrides old. priorize old -> `0 <= x <= 1` -> priorize new
    LEARNING_RATE = config["learningRate"]
    # How much we look to the future. immediate reward -> `0 <= x < 1` -> future rewards.
    DISCOUNT_FACTOR = config["discountFactor"]

    def __init__(self, low, high, number_of_actions):
        self.low = np.array(low)
        self.high = np.array(high)
        self.buckets = [self.NUMBER_OF_BUCKETS] * len(self.high)
        self.bucket_sizes = (self.high - self.low) / self.buckets
        # Initialize table with randoms for each bucket and action
        self.table = np.random.uniform(low=config["startLow"], high=config["startHigh"], size=(self.buckets + [number_of_actions]))


    def load_table(self, table):
        self.table = table

    def describe_table(self):
        print("------ QTable ------")
        print("Bucket space description", self.buckets)
        print("Bucket sizes", self.bucket_sizes)
        print("Table info", self.table.shape)

    def get_discrete(self, state):
        discrete = (state - self.low) / self.bucket_sizes
        return tuple(discrete.astype(np.int))

    def get_best_action(self, current_state, epsilon):
        if np.random.rand() > epsilon :
            return np.argmax(self.table[current_state])
        else:
            return np.random.randint(0, len(self.table[current_state]))

    def get_best_action_without_random(self, current_state):
        return np.argmax(self.table[current_state])
    

    def update_q(self, current_state, action, new_state, reward):
        max_future_q = np.max(self.table[new_state])
        current_q_index = current_state + (action, )
        current_q = self.table[current_q_index]
        new_q = (1 - self.LEARNING_RATE) * current_q + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * max_future_q)
        self.table[current_q_index] = new_q

    def new_epsilon(self, epsilon):
        self.EPSILON = epsilon
