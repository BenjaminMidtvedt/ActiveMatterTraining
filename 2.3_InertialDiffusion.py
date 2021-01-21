from ampy import stochastics, statistics

import numpy as np
import matplotlib.pyplot as plt

num_traces = 100


m = 1e-18
R = 1e-6
gamma = 6 * np.pi * R * 1e-3
tau = m / gamma
dt = 0.1 * tau
time = tau * 100

inertial = [True, False]
colors = ["r", "g", "b", "o", "k", "m"]

ax_short = plt.subplot(2, 2, 1)
ax_long = plt.subplot(2, 2, 2)

ax_low = plt.subplot(2, 1, 2)
ax_low.set_yscale("log")
ax_low.set_xscale("log")

for idx, inert in enumerate(inertial):

    steps = int(time / dt)
    msd = []
    t = np.linspace(0, time, steps)
    for step in range(num_traces):
        # We seed the random generator so that the inertial and non-inertial generate the same sequence
        np.random.seed(1 + step)
        trace = (
            stochastics.Diffusion(steps, m=m, gamma=gamma, dt=dt, inertial=inert) * 1e9
        )

        # Calcualte and append the msd
        msd.append(statistics.MSD(np.expand_dims(trace, axis=-1)))

        # Only show first trace
        if step == 0:
            ax_short.plot(t[:100], trace[:100], c=colors[idx])
            ax_long.plot(t, trace, c=colors[idx], linewidth=0.5)

    msd = np.mean(msd, axis=0)

    ax_low.plot(t[5:], msd[5:], c=colors[idx], alpha=0.5)


plt.show()