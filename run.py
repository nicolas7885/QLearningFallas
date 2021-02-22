import gym
from environment import Environment
import matplotlib.pyplot as plotter
import yaml
import numpy as np

config = yaml.safe_load(open("config.yaml"))
# TODO: use logger with log level
DEBUG = config["run"]["debug"]
RENDER = config["run"]["render"]
EPISODES = config["run"]["episodes"]
ALIVE_EVERY = config["run"]["aliveEvery"]
epsilon = config["run"]["epsilon"]
EPSILON_DECAY_START = 1
EPSILON_DECAY_END = EPISODES // 2
EPSILON_DECAY_VALUE = epsilon / (EPSILON_DECAY_END - EPSILON_DECAY_START)

rewards = []
aggregates = {'ep': [], 'avg': [], 'min': [], 'max': []}



env = Environment(config["environment"]["name"])
if DEBUG : env.describe_environment()
for episode in range(EPISODES):
    if DEBUG and not episode % ALIVE_EVERY :
        print(f"------ Starting Episode {episode}------")
    if RENDER:
        env.set_render(episode % config["run"]["renderEvery"] == 0)
    result = env.run_episode(epsilon)
    if EPSILON_DECAY_END > episode > EPSILON_DECAY_START :
        epsilon -= EPSILON_DECAY_VALUE
    rewards.append(result)
    if not episode % ALIVE_EVERY :
        last_section = rewards[-ALIVE_EVERY:]
        average = sum(last_section) / len(last_section)
        aggregates['ep'].append(episode)
        aggregates['avg'].append(average)
        aggregates['min'].append(min(last_section))
        aggregates['max'].append(max(last_section))
        plotter.plot(aggregates['ep'], aggregates['avg'], label='avg', color = 'g')
        plotter.plot(aggregates['ep'], aggregates['min'], label='min', color = 'r')
        plotter.plot(aggregates['ep'], aggregates['max'], label='max', color = 'b')
        handles, labels = plotter.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plotter.legend(by_label.values(), by_label.keys())
        plotter.savefig('plot.png')
        np.save(f"qtables/{episode}-qtable.npy", env.get_qTable)

env.close()


plotter.show()