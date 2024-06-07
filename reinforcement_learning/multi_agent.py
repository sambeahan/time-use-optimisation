from stable_baselines3 import A2C, PPO, DQN
from environments import *
from objective_functions import calc_outcomes
import numpy as np
import time
import json
from pathlib import Path

PARENT_DIR = Path(__file__).parent.parent
ALGORITHM = "PPO"  # also need to change on line 19

RUNS = 100

env = TimeUseEnv()

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
models = {}

for agent in agents:
    models[agent] = PPO.load(
        f"reinforcement_learning/models/static-{agent}-{ALGORITHM}-1-0"
    )

time_totals = {"Sleep": 0, "Sedentary": 0, "Active": 0}
time_use_vals = {"Sleep": [], "Sedentary": [], "Active": []}

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

agent_choices = {}
for agent in agents:
    agent_choices[agent] = [0, 0, 0]

reward = 0
reward_vals = []
turns = 0

start = time.time()

for i in range(RUNS):
    print("Run:", i)
    obs, info = env.reset()
    while True:
        prev_obs = [0, 0, 0]
        for agent, model in models.items():
            # print(agent)
            action, _states = model.predict(obs)
            # print(action)
            obs, rewards, dones, truncated, info = env.step(action, agent, False)

            reward += rewards
            reward_vals.append(rewards)
            turns += 1

            valid_action = False
            for j, time_spent in enumerate(obs):
                if time_spent != prev_obs[j]:
                    valid_action = True

            if valid_action:
                agent_choices[agent][action] += 0.1
            # print(obs)
            if dones:
                break
        if dones:
            break

    for j, time_used in enumerate(time_totals):
        time_totals[time_used] += obs[j]
        time_use_vals[time_used].append(obs[j])

    results = calc_outcomes(obs[0], obs[1], obs[2])

    for outcome, value in results.items():
        health_totals[outcome] += value
        health_vals[outcome].append(value)

end = time.time()

print("\nAverage time use:")
for label, time_used in time_totals.items():
    print(label + ":", time_used / RUNS)


print("\nAverage health outcomes:")
for outcome, value in health_totals.items():
    print(outcome + ":", value / RUNS)


print("\nStandard deviation:")
for label, times in time_use_vals.items():
    print(label + ":", np.std(times))

for outcome, values in health_vals.items():
    values = np.array(values)
    print(outcome + ":", np.std(values))

print("\nAgent choices")
for agent in agent_choices:
    print(agent + ":", [time_spent / RUNS for time_spent in agent_choices[agent]])

print("\nAverage reward:", reward / turns)
print("Min:", min(reward_vals))
print("Max:", max(reward_vals))
print("Std:", np.std(np.array(reward_vals)))

print("\nRuntime:", (end - start) / RUNS)

with open(
    Path(PARENT_DIR, "reinforcement_learning", "results", f"{ALGORITHM}_reward.json"),
    "w",
) as f:
    json.dump(reward_vals, f)
