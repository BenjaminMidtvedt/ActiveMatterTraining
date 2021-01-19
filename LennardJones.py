from ampy import ode

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spdist

m = 1
dt = 1 / 100
T = 50
eps = 1
sigma = 0.5
num_particles = 20

x_size = 20
y_size = 20

np.random.seed(0)


def LJ(r):
    return 12 * eps / r * (2 * (sigma / r) ** 12 - (sigma / r) ** 6)


def acceleration(x):
    pos = x[0]
    r = spdist.pdist(pos, "euclidean")

    particle_force_amplitude = spdist.squareform(LJ(r))
    r = spdist.squareform(r)

    accel = []
    for i in range(len(pos)):
        total_force = 0
        for j in range(len(pos)):
            if i == j:
                continue

            direction = pos[i] - pos[j]

            total_force += particle_force_amplitude[i, j] * direction / r[i, j]

        # Boundaries (TODO: Vectorize)
        total_force += LJ(np.abs(pos[i][0])) * np.array((1, 0))
        total_force += LJ(np.abs(pos[i][1])) * np.array((0, 1))
        total_force += LJ(np.abs(x_size - pos[i][0])) * np.array((-1, 0))
        total_force += LJ(np.abs(y_size - pos[i][1])) * np.array((0, -1))

        accel.append(total_force / m)

    return np.array(accel)


acceleration([np.random.rand(10, 2)])

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
    p0 = []

    # Create an evenly spaced grid of particles as a safe initialization
    for y0 in range(5, y_size, 3):
        for x0 in range(5, y_size, 3):
            p0.append((x0, y0))

            if len(p0) == num_particles:
                break
        if len(p0) == num_particles:
            break

    # Calculate a random angle as the starting velocity
    ang0 = np.random.rand(num_particles) * np.pi * 2
    v0 = [*zip(np.cos(ang0), np.sin(ang0))]
    x0 = [np.array(p0), np.array(v0) * 2]

    i = 0
    for pos, vel in solver.iter(x0, T):

        # Show every 5 steps
        if i % 10 == 0:
            plt.clf()
            plt.scatter(pos[:, 0], pos[:, 1], s=200)
            plt.xlim(0, x_size)
            plt.ylim(0, y_size)
            plt.ion()
            plt.show()
            plt.pause(0.01)

        i = i + 1


plt.show()