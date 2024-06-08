# importing package
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PARENT_DIR = Path(__file__).resolve().parent.parent
GRAPH_PATH = Path(PARENT_DIR, "graph_generation", "graphs", "dqn_choice.pdf")

BAR_WIDTH = 0.5

fig, ax = plt.subplots(figsize=(8, 5))
ax.grid(axis="y", zorder=0)

# create data
x = ["Sleep", "Sedentary", "Active"]
y1 = np.array([4, 1, 0.5])
y2 = np.array([2.94, 0.47, 0.29])
y3 = np.array([0.07, 3.57, 0.06])
y4 = np.array([0.06, 1.85, 1.78])
y5 = np.array([0.06, 1.38, 2.25])
y6 = np.array([2.30, 0.55, 0.85])

# plot bars in stack manner
ax.bar(x, y1, width=BAR_WIDTH, color="#a6a6a6", label="Lower bound", zorder=3)
ax.bar(x, y2, width=BAR_WIDTH, bottom=y1, label="Stress", zorder=3)
ax.bar(x, y3, width=BAR_WIDTH, bottom=y1 + y2, label="Resting heart rate", zorder=3)
ax.bar(
    x,
    y4,
    width=BAR_WIDTH,
    bottom=y1 + y2 + y3,
    label="Systolic blood pressure",
    zorder=3,
)
ax.bar(
    x,
    y5,
    width=BAR_WIDTH,
    bottom=y1 + y2 + y3 + y4,
    label="Diastolic blood pressure",
    zorder=3,
)
ax.bar(x, y6, width=BAR_WIDTH, bottom=y1 + y2 + y3 + y4 + y5, label="BMI", zorder=3)

plt.legend()
plt.xlabel("Activity")
plt.ylabel("Time (hours)")
plt.title("Time use chosen by each DQN RL agent")
# plt.show()
plt.savefig(GRAPH_PATH)

# print(np.sum(y1) + np.sum(y2) + np.sum(y3) + np.sum(y4) + np.sum(y5) + np.sum(y6))
