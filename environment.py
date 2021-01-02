import gym
from qTable import QTable
import yaml
import time
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
# TODO avoid needing this debug here?
DEBUG = config["run"]["debug"]
class Environment:
    def __init__(self, environmentName, iterations_per_episode):
        self.env = gym.make(environmentName)
        self.iterations_in_episode = iterations_per_episode
        # self.qTable = QTable(self.env)
        # if DEBUG: self.qTable.describe_table()
        self.render = False

    def describe_environment(self):
        print("------ Environment Description ------")
        print(self.env)
        print("Action space", self.env.action_space)
        print("Observation space", self.env.observation_space)
        print("Observation space High", self.env.observation_space.high)
        print("Observation space Low", self.env.observation_space.low)

    def run_episode(self, epsilon):
        old_state = self.env.reset()
        # current_discrete_state = self.qTable.get_discrete(self.env.reset())
        print("Bytes: [Tiempo: 90 91 Misc: 95 Bola: 99 101]")
        for iteration in range(self.iterations_in_episode):
            if self.render : 
                self.env.render()
                time.sleep(0.050)
            # action = self.qTable.get_best_action(current_discrete_state, epsilon)
            action = 0 if iteration < 13 else self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            bytes_changed = set()
            for byte in range(len(old_state)):
                if old_state[byte] != new_state[byte] and (not byte in [57, 83, 103, 72, 70, 90, 91, 95, 99, 101, 105]):
                    bytes_changed.add(byte)
            print(f"Bytes: [lives: {new_state[57]} Plataforma: {new_state[70]} {new_state[72]} Bola: {new_state[99]} {new_state[101]}  other: {new_state[77]} {new_state[84]}]")
            if len(bytes_changed):
                print("Bytes changed in iteration: ", iteration, bytes_changed)        
            # print(f"Reward: {reward} Done: {done}")
            # new_discrete_state = self.qTable.get_discrete(new_state)
            if done:
                break 
                # self.qTable.update_q(current_discrete_state, action, new_discrete_state, reward)
            # elif new_state[0] >= self.env.goal_position :
                # if DEBUG: print("Won on iteration", iteration)
                # self.qTable.mark_state_as_win(current_discrete_state, action)
                # break
            # current_discrete_state = new_discrete_state
            old_state = new_state
    
    def close(self):
        if DEBUG: print("------ Done ------")
        self.env.close()

    def set_render(self, shouldRender):
        self.render = shouldRender
