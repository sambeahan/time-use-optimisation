import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
GRAPH_PATH = Path(PARENT_DIR, "graph_generation", "graphs", "a_sed_work.pdf")

# set width of bar
BAR_WIDTH = 0.2
fig, ax = plt.subplots(figsize=(6, 5))
ax.grid(axis="y", zorder=0)

# set height of bar
stat = [8.80, 12.74, 2.46]
dyn = [9.06, 12.21, 2.74]
lower_bound = [4, 7.5, 0.5]
upper_bound = [11, 18, 12]

# Set position of bar on X axis
br1 = np.arange(len(stat))
br2 = [x + BAR_WIDTH for x in br1]
br3 = [x + BAR_WIDTH / 2 for x in br1]

# Make the plot
ax.bar(br1, stat, width=BAR_WIDTH, label="Static", zorder=3)
ax.bar(br2, dyn, width=BAR_WIDTH, label="Dynamic", zorder=3)
ax.bar(
    br3,
    lower_bound,
    width=BAR_WIDTH * 2,
    label="Lower bound",
    color="#a6a6a6",
    zorder=3,
)

for i, bound in enumerate(upper_bound):
    if i == 0:
        ax.plot(
            [br3[i] - BAR_WIDTH * 1.5, br3[i] + BAR_WIDTH * 1.5],
            [bound, bound],
            "--",
            color="gray",
            label="Uper bound",
        )
    else:
        ax.plot(
            [br3[i] - BAR_WIDTH * 1.5, br3[i] + BAR_WIDTH * 1.5],
            [bound, bound],
            "--",
            color="gray",
        )


# Adding Xticks
plt.xlabel("Activity")
plt.ylabel("Time (hours)")
plt.xticks(
    [r + BAR_WIDTH * 0.5 for r in range(len(stat))], ["Sleep", "Sedentary", "Active"]
)

plt.legend()
plt.title("Sedentary work day time use (Week A)")
# plt.show()
plt.savefig(GRAPH_PATH)
