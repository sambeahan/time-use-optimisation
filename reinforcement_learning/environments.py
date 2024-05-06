import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import objective_functions


class TimeUseEnv(gym.Env):
    def __init__(self) -> None:
        # define bounds
        self.max_sleep = 24.0
        self.min_sleep = 0.0
        self.max_sedentary = 24.0
        self.min_sedentary = 0.0
        self.max_active = 24.0
        self.min_active = 0.0

        self.action_space = spaces.Discrete(3)

        obs_low = np.array([0.1, 0.1, 0.1])
        obs_high = np.array([self.max_sleep, self.max_sedentary, self.max_active])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)

        self.current_obs = None
        self.time_left = 23.7

    def reset(self, seed=None, options=None):
        self.current_obs = np.array([np.float32(0.1), np.float32(0.1), np.float32(0.1)])

        return self.current_obs, {}

    def step(self, action, agent):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = 0
        post_obj = 0
        if agent == "stress":
            current_obj = objective_functions.calc_stress(self.current_obs.tolist())
            post_obj = objective_functions.calc_stress(next_obs.tolist())
        elif agent == "hr":
            current_obj = objective_functions.calc_hr(self.current_obs.tolist())
            post_obj = objective_functions.calc_hr(next_obs.tolist())
        elif agent == "sbp":
            current_obj = objective_functions.calc_sbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_sbp(next_obs.tolist())
        elif agent == "dbp":
            current_obj = objective_functions.calc_dbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_dbp(next_obs.tolist())
        elif agent == "bmi":
            current_obj = objective_functions.calc_bmi(self.current_obs.tolist())
            post_obj = objective_functions.calc_bmi(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return


class StressEnv(TimeUseEnv):
    def step(self, action):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = objective_functions.calc_stress(self.current_obs.tolist())
        post_obj = objective_functions.calc_stress(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}


class HREnv(TimeUseEnv):
    def step(self, action):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = objective_functions.calc_hr(self.current_obs.tolist())
        post_obj = objective_functions.calc_hr(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}


class SBPEnv(TimeUseEnv):
    def step(self, action):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = objective_functions.calc_sbp(self.current_obs.tolist())
        post_obj = objective_functions.calc_sbp(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}


class DBPEnv(TimeUseEnv):
    def step(self, action):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = objective_functions.calc_dbp(self.current_obs.tolist())
        post_obj = objective_functions.calc_dbp(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}


class BMIEnv(TimeUseEnv):
    def step(self, action):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        current_obj = objective_functions.calc_bmi(self.current_obs.tolist())
        post_obj = objective_functions.calc_bmi(next_obs.tolist())

        reward = -1 * post_obj

        self.time_left -= 0.1

        done = False
        if self.time_left <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {}


if __name__ == "__main__":
    env = TimeUseEnv()
    obs = env.reset()
    while True:
        action = random.randint(0, 2)
        obs, r, done, _, _ = env.step(action, "stress")
        print(obs)
        if done:
            break
