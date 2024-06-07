import time
import sys
import os
from pathlib import Path
from stable_baselines3 import A2C
from environments import *

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from objective_functions import calc_outcomes

HEALTH_OUTCOME = "stress"
ALGORITHM = "A2C"  # also need to change on line 20
MODEL_VERSION = "1-0"
TRAINING_METHOD = "static"

PARENT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(PARENT_DIR, "reinforcement_learning", "models")

env = TimeUseEnv()
model = A2C.load(
    (Path(MODEL_DIR, f"{TRAINING_METHOD}-{HEALTH_OUTCOME}-{ALGORITHM}-{MODEL_VERSION}"))
)

start = time.time()

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, truncated, info = env.step(action, "stress", False)
    print(obs)
    print(dones)
    if dones:
        break

    if time.time() - start > 30:
        raise TimeoutError(
            "Agent ran out of time to complete the optimisation scenario. This has likely occured due to it continuously seelcting invalid actions which have already reached their upper bounds."
        )

results = calc_outcomes(obs[0], obs[1], obs[2])

print("\nHealth outcomes:")
for result, value in results.items():
    print(f"{result}: {results[result]}")
