# AI-Based Time Use Optimisation for Improving Health Outcomes

This is a repository for optimising daily time use to achieve better health outcomes. To install the dependecies for this repository, run the following command:

```
pip install -r requirements.txt
```

The evolutionary optimisation is contained in two folders: `evo_single` for optimising a single objective (stress) with a genetic algorithm, and `evo_multi` for multi-objective optimisation using the NSGA-II algorithm

`reinforcement_learning` contains the reinforcement learning approach for optimisation.

The `evo_multi` and `reinforcement_learning` folders both contain a seperate README for further instructions.

`objective_functions.py` - objective functions for health outcomes for optimisation algorithms to use
