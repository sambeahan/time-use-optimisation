from stable_baselines3.common.env_checker import check_env
from environments import *

env = StressEnv()
check_env(env)

env = HREnv()
check_env(env)

env = SBPEnv()
check_env(env)

env = DBPEnv()
check_env(env)

env = BMIEnv()
check_env(env)
