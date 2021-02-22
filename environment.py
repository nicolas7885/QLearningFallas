import gym
from qTable import QTable
import yaml
import time
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
# TODO avoid needing this debug here?
DEBUG = config["run"]["debug"]
RENDER_DELAY = config["environment"]["renderDelay"]

class Environment:
    BALL_Y_POSITION_BYTE = 101
    BALL_FELL_POSITION = 178

    def __init__(self, environmentName):
        self.env = gym.make(environmentName)
        self.qTable = QTable(
            self.get_position_bytes(self.env.observation_space.low),
            self.get_position_bytes(self.env.observation_space.high),
            self.env.action_space.n
        )
        if DEBUG: self.qTable.describe_table()
        self.render = False

    def get_position_bytes(self, state):
        return [state[72], state[99], state[self.BALL_Y_POSITION_BYTE]]

    def get_reward(self, state):
        # Penalyze ball below platform
        return -10 if state[self.BALL_Y_POSITION_BYTE] >= self.BALL_FELL_POSITION or state[self.BALL_Y_POSITION_BYTE] == 0 else 0

    def get_score(self, state):
        return state[77]

    def describe_environment(self):
        print("------ Environment Description ------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)

    def run_episode(self, epsilon):
        current_discrete_state = self.qTable.get_discrete(self.get_position_bytes(self.env.reset()))
        done = False
        # Force start game to avoid learning to NOT play
        self.env.step(1)
        while not done:
            if self.render : 
                self.env.render()
                time.sleep(RENDER_DELAY)
            action = self.qTable.get_best_action(current_discrete_state, epsilon)
            new_state, hit, done, _ = self.env.step(action)
            new_discrete_state = self.qTable.get_discrete(self.get_position_bytes(new_state))
            reward = self.get_reward(new_state)
            if not done:
                self.qTable.update_q(current_discrete_state, action, new_discrete_state, reward)
            current_discrete_state = new_discrete_state
        return self.get_score(new_state)

    def close(self):
        if DEBUG: print("------ Done ------")
        self.env.close()

    def get_qTable():
        return self.qTable.table

    def set_render(self, shouldRender):
        self.render = shouldRender
