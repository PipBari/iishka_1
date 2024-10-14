import numpy as np

from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
