import time
from stable_baselines3 import A2C, PPO, DQN
from environments import *

STARTING_HOURS = 5.5
TRAINING_EPISODES = 1000

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
envs = [StressEnv(), HREnv(), SBPEnv(), DBPEnv(), BMIEnv()]

start = time.time()

for i, agent in enumerate(agents):
    env = envs[i]
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=(240 - STARTING_HOURS * 10) * TRAINING_EPISODES)
    model.save(f"reinforcement_learning/models/static-{agent}-DQN-1-1")

end = time.time()

print("Training time:", end - start)
