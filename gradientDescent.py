# Реализует алгоритм градиентного спуска для обновления параметров модели и минимизации функции стоимости
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)  # Количество обучающих примеров
    J_history = []  # Список для хранения значений функции стоимости

    # Итерации градиентного спуска
    for i in range(num_iters):
        h = []  # Список для предсказаний h

        # Вычисление предсказаний h для каждого примера
        for j in range(m):
            h_j = 0  # Начальное значение для предсказания h[j]
            for k in range(len(theta)):
                h_j += X[j][k] * theta[k]  # Суммируем произведения X[j][k] и theta[k]
            h.append(h_j)  # Добавляем предсказание в список h

        # Обновление параметров theta
        for k in range(len(theta)):
            sum_errors = 0  # Сумма ошибок для theta[k]
            for j in range(m):
                sum_errors += (h[j] - y[j]) * X[j][k]  # Суммируем ошибки
            theta[k] -= (alpha / m) * sum_errors  # Обновляем параметр theta[k] на основе средней ошибки

        # Сохранение стоимости после обновления параметров
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

# сделать доп файл для векторной реализации с возможностью выбора при запуске