from stable_baselines3 import A2C
from environments import *
from objective_functions import calc_outcomes

env = TimeUseEnv()
model = A2C.load("reinforcement_learning/models/stress-agent")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, truncated, info = env.step(action, "stress", False)
    print(obs)
    print(dones)
    if dones:
        break

results = calc_outcomes(obs[0], obs[1], obs[2])

print("\nHealth outcomes:")
for result, value in results.items():
    print(f"{result}: {results[result]}")
