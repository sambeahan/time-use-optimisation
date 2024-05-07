from stable_baselines3 import A2C
from environments import *

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
envs = [StressEnv(), HREnv(), SBPEnv(), DBPEnv(), BMIEnv()]

for i, agent in enumerate(agents):
    env = envs[i]
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=240 * 5)
    model.save(f"reinforcement_learning/models/{agent}-agent")
