import json
from pathlib import Path
import matplotlib.pyplot as plt

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(PARENT_DIR, "reinforcement_learning", "results")

with open(Path(DATA_DIR, "DQN_reward.json")) as f:
    data = json.load(f)

plt.plot(range(len(data)), data)
plt.show()
