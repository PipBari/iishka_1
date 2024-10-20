import numpy as np
from computeCost import computeCost

def gradientDescentVectorized(X, y, theta, alpha, num_iters):
    m = len(y)  # Количество примеров
    J_history = np.zeros(num_iters)  # Массив для стоимости

    for i in range(num_iters):
        h = X.dot(theta)  # Умножение матрицы X на вектор theta для получения предсказаний

        # Обновление theta: корректировка на основе ошибок
        theta -= (alpha / m) * (X.T.dot(h - y))  # Вычисление градиента и обновление параметров (используется транспонированная матрица (mn) (nm))

        J_history[i] = computeCost(X, y, theta)  # Сохранение стоимости

    return theta, J_history  # Возврат обновленных параметров и истории стоимости
