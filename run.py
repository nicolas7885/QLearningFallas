import gym
from environment import Environment
import yaml
config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
DEBUG = config["run"]["debug"]
RENDER = config["run"]["render"]

env = Environment(config["environment"]["name"], config["environment"]["iterations"])
if DEBUG : env.describe_environment()
for episode in range(config["run"]["episodes"]):
    if DEBUG and episode % config["run"]["aliveEvery"]  == 0 :
        print(f"------ Starting Episode {episode}------")
    if RENDER:
        env.set_render(episode % config["run"]["renderEvery"] == 0)
    env.run_episode()
env.close()
