from stable_baselines3 import A2C, PPO, DQN
from environments import *
from objective_functions import calc_outcomes
import numpy as np

env = TimeUseEnv()

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
models = {}

for agent in agents:
    models[agent] = PPO.load(f"reinforcement_learning/models/static-{agent}-PPO-1-0")

time_totals = {"Sleep": 0, "Sedentary": 0, "Active": 0}

health_totals = {
    "Stress Level": 0,
    "Resting Heart Rate": 0,
    "Systolic Blood Pressure": 0,
    "Diastolic Blood Pressure": 0,
    "BMI": 0,
}

health_vals = {
    "Stress Level": [],
    "Resting Heart Rate": [],
    "Systolic Blood Pressure": [],
    "Diastolic Blood Pressure": [],
    "BMI": [],
}

for i in range(100):
    print("Run:", i)
    obs, info = env.reset()
    while True:
        for agent, model in models.items():
            # print(agent)
            action, _states = model.predict(obs)
            # print(action)
            obs, rewards, dones, truncated, info = env.step(action, agent, False)
            # print(obs)
            if dones:
                break
        if dones:
            break

    for j, time in enumerate(time_totals):
        time_totals[time] += obs[j]

    results = calc_outcomes(obs[0], obs[1], obs[2])

    for outcome, value in results.items():
        health_totals[outcome] += value
        health_vals[outcome].append(value)

print("\nAverage time use:")
for label, time in time_totals.items():
    print(label + ":", time / 100)


print("\nAverage health outcomes:")
for outcome, value in health_totals.items():
    print(outcome + ":", value / 100)


print("\nStandard deviation:")
for outcome, values in health_vals.items():
    values = np.array(values)
    print(outcome + ":", np.std(values))
