from stable_baselines3 import A2C
from environments import *

env = SBPEnv()
model = A2C.load("reinforcement_learning/models/sbp-agent")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, truncated, info = env.step(action)
    print(obs)
    print(dones)
    if dones:
        break
