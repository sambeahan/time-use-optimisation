from stable_baselines3 import A2C
from environments import *
from objective_functions import calc_outcomes

env = TimeUseEnv()

agents = ["stress", "hr", "sbp", "dbp", "bmi"]
models = {}

for agent in agents:
    models[agent] = A2C.load(f"reinforcement_learning/models/static-{agent}-A2C-1-0")

time_totals = {"Sleep": 0, "Sedentary": 0, "Active": 0}

health_totals = {
    "Stress Level": 0,
    "Resting Heart Rate": 0,
    "Systolic Blood Pressure": 0,
    "Diastolic Blood Pressure": 0,
    "BMI": 0,
}

for i in range(100):
    print(i)
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

print("\nAverage time use:")
for label, time in time_totals.items():
    print(label + ":", time / 100)


print("\nAverage health outcomes:")
for outcome, value in health_totals.items():
    print(outcome + ":", value / 100)
