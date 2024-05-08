from stable_baselines3 import A2C
from environments import *
from objective_functions import calc_outcomes

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
        obs, rewards, dones, truncated, info = env.step(action, agent, False)
        print(obs)
        if dones:
            break
    if dones:
        break

results = calc_outcomes(obs[0], obs[1], obs[2])

print("\nHealth outcomes:")
for result in results:
    print(f"{result}: {results[result]}")
