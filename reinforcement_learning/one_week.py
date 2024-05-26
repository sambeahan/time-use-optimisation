import time
import numpy as np
from stable_baselines3 import A2C
from environments import *
from objective_functions import calc_outcomes


SEDENTARY_WORK = {"id": "sed_work", "lower": [4, 7.5, 0.5], "upper": [11, 18, 12]}
ACTIVE_WORK = {"id": "act_work", "lower": [4, 1, 7.5], "upper": [11, 12, 18]}
ACTIVE_WEEKEND = {"id": "act_wknd", "lower": [4, 1, 4], "upper": [12, 16, 16]}
RELAXING_WEEKEND = {"id": "rel_wknd", "lower": [8, 6, 0.5], "upper": [12, 18, 2]}
SOCIAL_WEEKEND = {"id": "soc_wknd", "lower": [4, 4, 1], "upper": [10, 18, 6]}

week_a = [SEDENTARY_WORK for _ in range(5)] + [ACTIVE_WEEKEND, RELAXING_WEEKEND]
week_b = [ACTIVE_WORK for _ in range(5)] + [RELAXING_WEEKEND, SOCIAL_WEEKEND]

WEEK = week_b
RUNS = 100

env = TimeUseEnv()

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
models = {}

for agent in agents:
    # models[agent] = A2C.load(f"reinforcement_learning/models/static-{agent}-A2C-1-0")
    models[agent] = A2C.load(f"reinforcement_learning/models/dynamic-{agent}-A2C-1-0")

day_id_list = [day["id"] for day in WEEK]
day_types = set(day_id_list)

day_type_freq = {}
for type_name in day_types:
    day_type_freq[type_name] = 0

for id in day_id_list:
    day_type_freq[id] += 1

print(day_types)

time_totals = {}

for type_name in day_types:
    time_totals[type_name] = [0, 0, 0]

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
for type_name in day_types:
    agent_choices[type_name] = {}
    for agent in agents:
        agent_choices[type_name][agent] = [0, 0, 0]

for i in range(RUNS):
    print("Run:", i)

    weekly_totals = {
        "Stress Level": 0,
        "Resting Heart Rate": 0,
        "Systolic Blood Pressure": 0,
        "Diastolic Blood Pressure": 0,
        "BMI": 0,
    }

    for j, day in enumerate(WEEK):
        print("Day:", j)
        obs, info = env.reset(lower_bound=day["lower"], upper_bound=day["upper"])
        while True:
            prev_obs = [0, 0, 0]
            for agent, model in models.items():
                # print(agent)

                action, _states = model.predict(obs)
                # print(action)

                obs, rewards, dones, truncated, info = env.step(action, agent, False)
                # print(obs)
                # print(info)

                valid_action = False
                for j, time_spent in enumerate(obs):
                    if time_spent != prev_obs[j]:
                        valid_action = True

                if valid_action:
                    agent_choices[day["id"]][agent][action] += 0.1

                if dones:
                    # print("Completed")
                    # print(obs)
                    # print(np.sum(obs))
                    break
            if dones:
                break

        for k, time_spent in enumerate(obs):
            time_totals[day["id"]][k] += time_spent

        results = calc_outcomes(obs[0], obs[1], obs[2])

        for outcome, value in results.items():
            health_totals[outcome] += value
            weekly_totals[outcome] += value
    for outcome, value in weekly_totals.items():
        health_vals[outcome].append(value / len(WEEK))

print("\nAverage time use:")
for type_name in day_types:
    print(type_name)
    for time_used in time_totals[type_name]:
        print(time_used / (RUNS * day_type_freq[type_name]))


print("\nAverage health outcomes:")
for outcome, value in health_totals.items():
    print(outcome + ":", value / (RUNS * len(WEEK)))


print("\nStandard deviation:")
for outcome, values in health_vals.items():
    values = np.array(values)
    print(outcome + ":", np.std(values))

print("\nAgent choices")
for type_name in day_types:
    print(type_name)
    for agent in agent_choices[type_name]:
        print(
            agent + ":",
            [
                time_spent / (RUNS * day_type_freq[type_name])
                for time_spent in agent_choices[type_name][agent]
            ],
        )

print(len(health_vals["Stress Level"]))
