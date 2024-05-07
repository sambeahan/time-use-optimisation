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
    # return current_obj - post_obj
    return -1 * post_obj


class TimeUseEnv(gym.Env):
    def __init__(self) -> None:
        # define bounds
        self.max_sleep = 12.0
        self.min_sleep = 4.0
        self.max_sedentary = 20.0
        self.min_sedentary = 2.0
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
        obs_high = np.array([self.max_sleep, self.max_sedentary, self.max_active])

        self.observation_space = spaces.Box(low=self.obs_start, high=obs_high)

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

    def step(self, action, agent):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        next_obs[action] += np.float32(0.1)

        reward = 0

        current_obj = 0
        post_obj = 0
        if agent == "stress":
            current_obj = objective_functions.calc_stress(self.current_obs.tolist())
            post_obj = objective_functions.calc_stress(next_obs.tolist())
            if action != self.stress_last_action:
                reward += STRESS_VAR_BONUS

            self.stress_last_action = action
        elif agent == "hr":
            current_obj = objective_functions.calc_hr(self.current_obs.tolist())
            post_obj = objective_functions.calc_hr(next_obs.tolist())
            if action != self.hr_last_action:
                reward += HR_VAR_BONUS

            self.hr_last_action = action
        elif agent == "sbp":
            current_obj = objective_functions.calc_sbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_sbp(next_obs.tolist())
            if action != self.sbp_last_action:
                reward += SBP_VAR_BONUS

            self.sbp_last_action = action
        elif agent == "dbp":
            current_obj = objective_functions.calc_dbp(self.current_obs.tolist())
            post_obj = objective_functions.calc_dbp(next_obs.tolist())
            if action != self.dbp_last_action:
                reward += DBP_VAR_BONUS

            self.dbp_last_action = action
        elif agent == "bmi":
            current_obj = objective_functions.calc_bmi(self.current_obs.tolist())
            post_obj = objective_functions.calc_bmi(next_obs.tolist())
            if action != self.bmi_last_action:
                reward += BMI_VAR_BONUS

            self.bmi_last_action = action

        reward += reward_add(current_obj, post_obj)

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

        reward = current_obj - post_obj

        self.time_left -= 0.1

        if action != self.stress_last_action:
            reward += STRESS_VAR_BONUS

        self.stress_last_action = action

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

        reward = current_obj - post_obj

        if action != self.hr_last_action:
            reward += HR_VAR_BONUS

        self.hr_last_action = action

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

        reward = current_obj - post_obj

        if action != self.sbp_last_action:
            reward += SBP_VAR_BONUS

        self.sbp_last_action = action

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

        reward = current_obj - post_obj

        if action != self.dbp_last_action:
            reward += DBP_VAR_BONUS

        self.dbp_last_action = action

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

        reward = current_obj - post_obj

        if action != self.bmi_last_action:
            reward += BMI_VAR_BONUS

        self.bmi_last_action = action

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
