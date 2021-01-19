from ampy import ode

import matplotlib.pyplot as plt
import numpy as np


m = 1
k = 1
dt = 1 / 20
T = 50


def acceleration(x):
    return -k / m * x[0]


tested_solvers = [ode.Euler, ode.LeapFrog]

# =========================================================
# START
# =========================================================

plt.figure(figsize=(15, 10))

num_solvers = len(tested_solvers)
t = np.arange(0, T, dt)

str_struct = "{0}: found frequency of {1:.3f}, expected {2:.3f}"

for idx, solver_class in enumerate(tested_solvers):

    solver = solver_class(acceleration, dt)

    x0 = [(1), (0)]
    pos, vel = zip(*[x for x in solver.iter(x0, T)])

    # Get system energy over time
    kinetic_energy = np.array(vel) ** 2 * m / 2
    potential_energy = np.array(pos) ** 2 * k / 2
    total_energy = kinetic_energy + potential_energy

    # Compare to analytical
    omega = np.sqrt(k / m)
    A = np.sqrt(x0[0] + (x0[1] / omega) ** 2)
    analytical = A * np.cos(omega * t)

    error = np.square(pos - analytical)

    # Get simulation frequency
    x_fft = np.abs(np.fft.fft(pos))

    # Search from  1 to ignore the zero component (probably not needed).
    a_max = np.argmax(x_fft[1:100]) + 1
    # plt.plot(x_fft[0:4])
    # plt.show()

    print(
        str_struct.format(
            solver_class.__name__, a_max / dt / len(pos), omega / 2 / np.pi
        )
    )

    plt.subplot(3, len(tested_solvers), 1 * len(tested_solvers) + idx - 1)
    plt.plot(t, pos)
    plt.title(solver_class.__name__)
    plt.ylabel("Position")

    plt.subplot(3, len(tested_solvers), 2 * len(tested_solvers) + idx - 1)
    plt.plot(t, total_energy)
    plt.ylabel("Energy")

    plt.subplot(3, len(tested_solvers), 3 * len(tested_solvers) + idx - 1)
    plt.plot(t, error)
    plt.ylabel("Squared error")

    plt.xlabel("Time")


plt.show()