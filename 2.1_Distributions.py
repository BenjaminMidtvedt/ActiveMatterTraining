from ampy import stochastics

import numpy as np
import matplotlib.pyplot as plt

num_values = 10000
num_traces = 1000

# The modes of the discrete distribution
modes = [
    (1, -np.sqrt(6) / 2),
    (1, -3 * np.sqrt(6) / 10),
    (2, 2 * np.sqrt(6) / 5),
]

# All distributions we will try
distributions = [
    stochastics.CoinFlip,
    np.random.randn,
    lambda x: stochastics.DiscreteDistribution(x, modes),
]
# The colors representing htose distributions
colors = ["r", "g", "b", "o", "k", "m"]

# Allocate matrix for storing endpoints
endpoints = np.zeros((len(distributions), num_traces))

# Create subplot axes
axes = [plt.subplot(2, 3, idx + 1) for idx in range(len(distributions))]

for step in range(num_traces):

    for idx, distribution in enumerate(distributions):
        trace = np.cumsum(distribution(num_values))
        endpoints[idx, step] = trace[-1]

        axes[idx].plot(trace, c=colors[idx], alpha=0.1, linewidth=0.5)
        axes[idx].set_ylim([-300, 300])

        if idx > 0:
            axes[idx].set_yticks([])


plt.subplot(2, 1, 2)
for idx, endpoint in enumerate(endpoints):
    plt.hist(endpoint, facecolor=colors[idx], alpha=0.2)

plt.show()