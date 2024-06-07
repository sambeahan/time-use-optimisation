# Reinforcement Learning Time Use Optimisation

## Running RL optimisation

*Note: it is recommended to use the 1-0 version of each agent*

To run a single RL agent on the single day problem, run the `single_agent.py` file. For multi objective optimisation with the multi-agent setup on the single day problem, run the `multi_agent.py` file. To run the multi-agent optimisation on the one week problem, use the `one_week.py` file. Within each of these files, the RL agents can be swapped in and out using the global variables at the start of the file.

## Training

To train a set of RL agents for each health outcome, use `training.py`. To switch between static and dynamic training, the reset function in `environments.py` needs to be changed accordingly.

## RL Environments

`environments.py` contains the environment for the time use problem, as well as the children environments for each health outcome. To test that the environments are still compatible with Gymnasium after any changes, use the `env_test.py` file. If no errors are produced, then the environments are compatible and can be used in other files.