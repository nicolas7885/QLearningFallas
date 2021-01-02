import gym
from qTable import QTable
import yaml
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
# TODO avoid needing this debug here?
DEBUG = config["run"]["debug"]
class Environment:
    def __init__(self, environmentName, iterations_per_episode):
        self.env = gym.make(environmentName)
        self.iterations_in_episode = iterations_per_episode
        self.qTable = QTable(self.env)
        if DEBUG: self.qTable.describe_table()
        self.render = False

    def describe_environment(self):
        print("------ Environment Description ------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)
        print("Goal", self.env.goal_position)

    def run_episode(self, epsilon):
        current_discrete_state = self.qTable.get_discrete(self.env.reset())
        for iteration in range(self.iterations_in_episode):
            if self.render : self.env.render()
            action = self.qTable.get_best_action(current_discrete_state, epsilon)
            new_state, reward, done, _ = self.env.step(action)
            new_discrete_state = self.qTable.get_discrete(new_state)
            if not done: 
                self.qTable.update_q(current_discrete_state, action, new_discrete_state, reward)
            elif new_state[0] >= self.env.goal_position :
                if DEBUG: print("Won on iteration", iteration)
                self.qTable.mark_state_as_win(current_discrete_state, action)
                break
            current_discrete_state = new_discrete_state

    def close(self):
        if DEBUG: print("------ Done ------")
        self.env.close()

    def set_render(self, shouldRender):
        self.render = shouldRender
