import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import objective_functions


STRESS_VAR_BONUS = 0
HR_VAR_BONUS = 0
SBP_VAR_BONUS = 0
DBP_VAR_BONUS = 0
BMI_VAR_BONUS = 0


def get_reward(current_obj, post_obj):
    # Individual action reward:
    return current_obj - post_obj

    # Cumulative reward:
    # return -1 * post_obj


class TimeUseEnv(gym.Env):
    def __init__(self) -> None:
        # define bounds
        self.lower_bound = np.array([np.float32(0.1) for _ in range(3)])

        self.upper_bound = np.array([24.0 for _ in range(3)])

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=self.lower_bound, high=self.upper_bound)

        self.stress_last_action = 0
        self.hr_last_action = 0
        self.sbp_last_action = 0
        self.dbp_last_action = 0
        self.bmi_last_action = 0

        self.current_obs = None
        self.time_left = 24 - np.sum(self.lower_bound)

    def reset(self, seed=None, options=None, lower_bound=None, upper_bound=None):
        if lower_bound is not None:
            self.lower_bound = np.array(lower_bound).astype(np.float32)
        else:
            # Static bounds:
            self.lower_bound = np.array(
                [np.float32(4.0), np.float32(1.0), np.float32(0.5)]
            )

            # Dynamic bounds:
            # self.lower_bound = np.array(
            #     [np.float32(random.randint(1, 120) / 10) for _ in range(3)]
            # )

        if upper_bound is not None:
            self.upper_bound = np.array(upper_bound).astype(np.float32)
        else:
            # Static bounds:
            self.upper_bound = np.array(
                [np.float32(12.0), np.float32(18.0), np.float32(12.0)]
            )

            # Dynamic bounds:
            # self.upper_bound = np.array(
            #     [np.float32(random.randint(121, 240) / 10) for _ in range(3)]
            # )

        while np.sum(self.lower_bound) >= 24.0:
            for i, time_used in enumerate(self.lower_bound):
                self.lower_bound[i] -= 0.1

        # print("Lower:", self.lower_bound)
        # print("Upper:", self.upper_bound)

        self.current_obs = np.copy(self.lower_bound)
        self.time_left = 24 - np.sum(self.lower_bound)

        # print(self.time_left)
        print("Reset")

        return self.current_obs, {}

    def step(self, action, agent, is_training=True):
        next_obs = np.zeros(3)

        for i, time in enumerate(self.current_obs):
            next_obs[i] = np.float32(time)

        # Only add action if below threshold
        valid_action = False
        if (
            round(next_obs[action] + np.float32(0.1), 1)
            <= round(self.upper_bound[action], 1)
            # or is_training
        ):
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

        if valid_action:
            reward += get_reward(current_obj, post_obj)
        else:
            reward = -100

        if valid_action:
            self.time_left -= 0.1

        done = False
        if round(self.time_left, 1) <= 0:
            done = True

        for i, time in enumerate(next_obs):
            self.current_obs[i] = np.float32(time)

        return self.current_obs, reward, done, False, {"valid": valid_action}

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
    obs = env.reset()
    obs = env.reset()
    while True:
        action = random.randint(0, 2)
        obs, r, done, _, _ = env.step(action, "stress")
        print(obs)
        if done:
            break
