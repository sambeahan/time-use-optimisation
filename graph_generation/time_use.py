import numpy as np
import matplotlib.pyplot as plt

# set width of bar
BAR_WIDTH = 0.1
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(axis="y", zorder=0)

# set height of bar
A2C = [8.95, 12.54, 2.61]
PPO = [9.34, 11.19, 3.57]
DQN = [9.52, 8.83, 5.74]
NSGA = [7.54, 14.2, 2.28]

# Set position of bar on X axis
br1 = np.arange(len(A2C))
br2 = [x + BAR_WIDTH for x in br1]
br3 = [x + BAR_WIDTH for x in br2]
br4 = [x + BAR_WIDTH for x in br3]

# Make the plot
ax.bar(br1, A2C, width=BAR_WIDTH, label="A2C", zorder=3)
ax.bar(br2, PPO, width=BAR_WIDTH, label="PPO", zorder=3)
ax.bar(br3, DQN, width=BAR_WIDTH, label="DQN", zorder=3)
ax.bar(br4, NSGA, width=BAR_WIDTH, label="NSGA-II", zorder=3)

# Adding Xticks
plt.xlabel("Activity")
plt.ylabel("Time (hours)")
plt.xticks(
    [r + BAR_WIDTH * 1.5 for r in range(len(A2C))], ["Sleep", "Sedentary", "Active"]
)

plt.legend()
plt.title("Average time use chosen by each algorithm")
# plt.show()
plt.savefig("graphs/time_use.pdf")
