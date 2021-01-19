from ampy import ode

import matplotlib.pyplot as plt
import numpy as np


m = 1
k = 1
dt = 1 / 20
T = 1000

d0 = 1
l0 = 0.5 * d0
A = 0.2 * d0
omega = 2 * np.pi / 25

subproblem = "c"  # "a" "b" or "c"


def acceleration(x):
    a = []
    for i, p in enumerate(x[0]):

        # The first particle in the Spring
        if i == 0:
            a.append(-p * (omega) ** 2)

        # The last particle in the string
        elif i == len(x[0]) - 1:

            if subproblem == "a":
                a.append(0)

            elif subproblem == "b":
                u1 = x[0][i - 1] - p
                dL1 = np.sqrt(u1 ** 2 + d0 ** 2)
                a.append((dL1 - l0) / dL1 * u1 * k / m)

            elif subproblem == "c":
                a.append(-p * (omega * 2) ** 2)

        # The middle particles
        else:
            u1 = x[0][i - 1] - p
            u2 = x[0][i + 1] - p
            dL1 = np.sqrt(u1 ** 2 + d0 ** 2)
            dL2 = np.sqrt(u2 ** 2 + d0 ** 2)

            a.append(((dL1 - l0) / dL1 * u1 + (dL2 - l0) / dL2 * u2) * k / m)

    return np.array(a)


tested_solvers = [ode.LeapFrog]

# =========================================================
# START
# =========================================================

plt.figure(figsize=(15, 10))

num_solvers = len(tested_solvers)
t = np.arange(0, T, dt)

str_struct = "{0}: found frequency of {1:.3f}, expected {2:.3f}"

for idx, solver_class in enumerate(tested_solvers):

    solver = solver_class(acceleration, dt)

    x0 = [np.zeros(120), np.zeros(120)]
    x0[0][0] = A
    if subproblem == "c":
        x0[0][-1] = A
    i = 0
    for pos, vel in solver.iter(x0, T):
        if i % 20 == 0:
            plt.clf()
            plt.scatter(np.arange(len(pos)), pos)
            plt.ylim(-2, 2)
            plt.ion()
            plt.show()
            plt.pause(0.01)

        i = i + 1
plt.show()