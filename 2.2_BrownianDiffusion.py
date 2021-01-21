from ampy import stochastics, statistics

import numpy as np
import matplotlib.pyplot as plt

num_traces = 1000

# Some reasonable but random constants
m = 1e-12
gamma = 1e-9
time = 20

dts = [0.01, 0.05, 0.1]

colors = ["r", "g", "b", "o", "k", "m"]

axes = [plt.subplot(2, len(dts), idx + 1) for idx in range(len(dts))]

ax_low = plt.subplot(2, 1, 2)

for idx, dt in enumerate(dts):

    steps = int(time / dt)
    msd = []
    t = np.linspace(0, time, steps)
    for step in range(num_traces):

        trace = stochastics.Diffusion(steps, m=m, gamma=gamma, dt=dt)

        # Calcualte and append the msd
        msd.append(statistics.MSD(np.expand_dims(trace, axis=-1)))

        axes[idx].plot(t, trace, c=colors[idx], alpha=0.1, linewidth=0.5)
        axes[idx].set_ylim([-5e-5, 5e-5])

        if idx > 0:
            axes[idx].set_yticks([])

    msd = np.mean(msd, axis=0)

    ax_low.plot(t, msd, c=colors[idx], alpha=0.5)
plt.show()