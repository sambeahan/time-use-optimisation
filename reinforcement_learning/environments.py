import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import objective_functions


STRESS_VAR_BONUS = 5
HR_VAR_BONUS = 10
SBP_VAR_BONUS = 100
DBP_VAR_BONUS = 100
BMI_VAR_BONUS = 10


def reward_add(current_obj, post_obj):
    # Individual action reward:
    return current_obj - post_obj

    # Cumulative reward:
    # return -1 * post_obj


class TimeUseEnv(gym.Env):
    def __init__(self) -> None:
        # define bounds
        self.max_sleep = 12.0
        self.min_sleep = 4.0
        self.max_sedentary = 18.0
        self.min_sedentary = 1.0
        self.max_active = 10.0
        self.min_active = 0.5

        self.action_space = spaces.Discrete(3)

        self.obs_start = np.array(
            [
                np.float32(self.min_sleep),
                np.float32(self.min_sedentary),
                np.float32(self.min_active),
            ]
        )
        self.obs_high = np.array([self.max_sleep, self.max_sedentary, self.max_active])

        self.observation_space = spaces.Box(low=self.obs_start, high=self.obs_high)

        self.stress_last_action = 0
        self.hr_last_action = 0
        self.sbp_last_action = 0
        self.dbp_last_action = 0
        self.bmi_last_action = 0

        self.current_obs = None
        self.time_left = 24 - np.sum(self.obs_start)

    def reset(self, seed=None, options=None):
        self.current_obs = self.obs_start

        return self.current_obs, {}

    def step(self, action, agent, is_training=True):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        # Only add action if below threshold
        valid_action = False
        if next_obs[action] + np.float32(0.1) <= self.obs_high[action] or is_training:
            next_obs[action] += np.float32(0.1)
            valid_action = True

        reward = 0

        current_obj = 0
        post_obj = 0

        if agent == "stress":
            current_obj = objective_functions.calc_stress(self.current_obs.tolist())
            post_obj = objective_functions.calc_stress(next_obs.tolist())
            if action != self.stress_last_action and valid_action:
                reward += STRESS_VAR_BONUS

            self.stress_last_action = action
        elif agent == "hr":
            current_obj = objective_functions.calc_hr(self.current_obs.tolist())
            post_obj = objective_functions.calc_hr(next_obs.tolist())
            if action != self.hr_last_action and valid_action:
                reward += HR_VAR_BONUS

            self.hr_last_action = action
        elif agent == "sbp":
            current_obj = objective_functions.calc_sbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_sbp(next_obs.tolist())
            if action != self.sbp_last_action and valid_action:
                reward += SBP_VAR_BONUS

            self.sbp_last_action = action
        elif agent == "dbp":
            current_obj = objective_functions.calc_dbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_dbp(next_obs.tolist())
            if action != self.dbp_last_action and valid_action:
                reward += DBP_VAR_BONUS

            self.dbp_last_action = action
        elif agent == "bmi":
            current_obj = objective_functions.calc_bmi(self.current_obs.tolist())
            post_obj = objective_functions.calc_bmi(next_obs.tolist())
            if action != self.bmi_last_action and valid_action:
                reward += BMI_VAR_BONUS

            self.bmi_last_action = action

        reward += reward_add(current_obj, post_obj)

        if valid_action:
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
        return super().step(action, "stress")


class HREnv(TimeUseEnv):
    def step(self, action):
        return super().step(action, "hr")


class SBPEnv(TimeUseEnv):
    def step(self, action):
        return super().step(action, "sbp")


class DBPEnv(TimeUseEnv):
    def step(self, action):
        return super().step(action, "dbp")


class BMIEnv(TimeUseEnv):
    def step(self, action):
        return super().step(action, "bmi")


if __name__ == "__main__":
    env = TimeUseEnv()
    obs = env.reset()
    while True:
        action = random.randint(0, 2)
        obs, r, done, _, _ = env.step(action, "stress")
        print(obs)
        if done:
            break
