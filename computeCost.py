# Вычисляет функцию стоимости для модели, чтобы оценить качество предсказаний
import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    J = (1/(2*m)) * np.sum(np.square(X.dot(theta) - y))
    return J
