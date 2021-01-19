from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    def __init__(self, acceleration, dt=1 / 100):
        self.acceleration = acceleration
        self.dt = dt

    def iter(self, x0, T):
        """
        x0: list of array-like
            Initial conditions for each object. Each tuple indiciates a derivative of a higher order.
            E.G. [pos, vel, acc, ...]
        """
        t = 0

        # Precondition as arraylike to save memory copies
        x = [np.array(d) for d in x0]

        while t < T - self.dt / 2:
            t = t + self.dt
            x = self.update(x)
            yield [*x]

    @abstractmethod
    def update(self, x):
        pass


class Euler(Solver):
    def update(self, x):
        a = self.acceleration(x)

        x[0] = x[0] + x[1] * self.dt
        x[1] = x[1] + a * self.dt
        return x


class LeapFrog(Solver):
    def update(self, x):

        # Update half-step
        x[0] = x[0] + x[1] * self.dt / 2

        # Acceleration with updated half-step
        a = self.acceleration(x)

        x[1] = x[1] + a * self.dt

        # Update second half-step
        x[0] = x[0] + x[1] * self.dt / 2
        return x