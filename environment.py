import gym

ITERATIONS_IN_EPISODE = 100
DEBUG = True
RENDER = False

def describe_environment(env):
    print("-------- Environment Description --------")
    print(env)
    print("Action space", env.action_space)
    print("Observation space", env.observation_space)
    print("Observation space High", env.observation_space.high)
    print("Observation space Low", env.observation_space.low)
    print("--------")

def run_episode(env):
    if DEBUG : print("-------- Starting Episode --------")
    env.reset()
    for _ in range(ITERATIONS_IN_EPISODE):
        # 0 lef 2 rig 1 nothing
        action = 2
        if RENDER : env.render()
        new_state, reward, done, _ = env.step(env.action_space.sample())
        print(new_state)


env = gym.make("MountainCar-v0")
if DEBUG : describe_environment(env)
run_episode(env)
env.close()