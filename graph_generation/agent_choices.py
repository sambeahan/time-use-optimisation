# importing package
import matplotlib.pyplot as plt
import numpy as np

BAR_WIDTH = 0.5

fig, ax = plt.subplots(figsize=(8, 5))
ax.grid(axis="y", zorder=0)

# create data
x = ["Sleep", "Sedentary", "Active"]
y1 = np.array([4, 1, 0.5])
y2 = np.array([3.33, 0.17, 0.30])
y3 = np.array([1.61, 2.08, 0])
y4 = np.array([0, 3.70, 0])
y5 = np.array([0, 3.70, 0])
y6 = np.array([0, 1.88, 1.81])

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
plt.title("Time use chosen by each A2C RL agent")
# plt.show()
plt.savefig("graphs/a2c_choice.pdf")
