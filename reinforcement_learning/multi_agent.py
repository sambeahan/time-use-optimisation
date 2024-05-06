from stable_baselines3 import A2C
from environments import *

env = TimeUseEnv()

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
models = {}

for agent in agents:
    models[agent] = A2C.load(f"reinforcement_learning/models/{agent}-agent")

obs, info = env.reset()
while True:
    for agent, model in models.items():
        print(agent)
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, truncated, info = env.step(action, agent)
        print(obs)
        if dones:
            break
    if dones:
        break
