import gym
from environment import Environment
import yaml
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
DEBUG = config["run"]["debug"]
RENDER = config["run"]["render"]
EPISODES = config["run"]["episodes"]
epsilon = config["run"]["epsilon"]
EPSILON_DECAY_START = 1
EPSILON_DECAY_END = EPISODES // 2
EPSILON_DECAY_VALUE = epsilon / (EPSILON_DECAY_END - EPSILON_DECAY_START)

env = Environment(config["environment"]["name"])
if DEBUG : env.describe_environment()
for episode in range(EPISODES):
    if DEBUG and episode % config["run"]["aliveEvery"]  == 0 :
        print(f"------ Starting Episode {episode}------")
    if RENDER:
        env.set_render(episode % config["run"]["renderEvery"] == 0)
    env.run_episode(epsilon)
    if EPSILON_DECAY_END > episode > EPSILON_DECAY_START :
        epsilon -= EPSILON_DECAY_VALUE
env.close()
