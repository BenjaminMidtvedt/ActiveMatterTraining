import numpy as np

Kb = 1.38e-23


def DiscreteDistribution(num_values, modes):
    """
    Modes should be in the form
    [(weight, value), (weight, value), ...]
    where the probability of each mode is weight / sum(weights)
    """

    modes = np.array(modes)

    probabilities = modes[:, 0] / np.sum(modes[:, 0])

    return np.random.choice(modes[:, 1], num_values, p=probabilities)


def CoinFlip(num_values):
    return DiscreteDistribution(num_values, [(1, -1), (1, 1)])


def Diffusion(
    num_values,
    m=1,
    gamma=1,
    T=300,
    dt=1 / 60,
    random_generator=np.random.randn,
    inertial="auto",
):

    tau = gamma / m

    Wt = random_generator(num_values)

    T_coef = 2 * Kb * T * dt
    # print(np.sqrt(T_coef / gamma))
    if inertial != False and (tau * dt < 100 or inertial == True):
        # Inertial

        coef_1 = (2 + dt * tau) / (1 + dt * tau)
        coef_2 = 1 / (1 + dt * tau)
        coef_3 = np.sqrt(T_coef * gamma) / (m * (1 + dt * tau)) * dt

        results = np.zeros((num_values + 2,))
        for step in range(2, num_values + 2):
            results[step] = (
                coef_1 * results[step - 1]
                - coef_2 * results[step - 2]
                + coef_3 * Wt[step - 2]
            )

            # print(results[step])

        return results[2:]
    else:
        # Non-inertial

        return np.cumsum(Wt) * np.sqrt(T_coef / gamma)
