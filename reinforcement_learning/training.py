import time
from pathlib import Path
from stable_baselines3 import A2C, PPO, DQN
from environments import *


TRAINING_EPISODES = 1000
ALGORITHM = "A2C"  # also need to change on line 22
MODEL_VERSION = "1-2"
TRAINING_METHOD = "static"  # change accordingly in the reset() function
# of the TimeUseEnv class in environments.py

STARTING_HOURS = 5.5  # the sum of the initial hours on each activity,
# used to calculate total timesteps

PARENT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(PARENT_DIR, "reinforcement_learning", "models")


agents = ["stress", "hr", "sbp", "dbp", "bmi"]
envs = [StressEnv(), HREnv(), SBPEnv(), DBPEnv(), BMIEnv()]

start = time.time()

for i, agent in enumerate(agents):
    env = envs[i]
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=(240 - STARTING_HOURS * 10) * TRAINING_EPISODES)
    model.save(
        Path(MODEL_DIR, f"{TRAINING_METHOD}-{agent}-{ALGORITHM}-{MODEL_VERSION}")
    )

end = time.time()

print("Training time:", end - start)
