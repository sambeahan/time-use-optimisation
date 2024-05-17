# Graph code adapted from: https://www.geeksforgeeks.org/bar-plot-in-matplotlib/

import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.1
fig = plt.subplots(figsize=(6, 4))

# set height of bar
A2C = [12, 30, 1]
PPO = [28, 6, 16]
DQN = [29, 3, 24]
NSGA = [29, 3, 24]

# Set position of bar on X axis
br1 = np.arange(len(A2C))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, A2C, width=barWidth, label="A2C")
plt.bar(br2, PPO, width=barWidth, label="PPO")
plt.bar(br3, DQN, width=barWidth, label="DQN")
plt.bar(br4, NSGA, width=barWidth, label="NSGA")

# Adding Xticks
plt.xlabel("Activity")
plt.ylabel("Time")
plt.xticks([r + barWidth for r in range(len(A2C))], ["Sleep", "Sedentary", "Active"])

plt.legend()
plt.show()
