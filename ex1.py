import numpy as np
import plotly.graph_objects as go

from gradientVector import gradientDescentVectorized
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from work import work

# Запросить у пользователя, какой вариант градиентного спуска использовать
option = input("Выберите вариант градиентного спуска (1 - обычный, 2 - векторный): ")

# Загрузка данных из файла
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]  # Количество автомобилей
y = data[:, 1]  # Прибыль СТО

# Визуализация данных
plotData(X, y)

# Настройка параметров
m = len(y)  # Количество обучающих примеров
X = np.hstack((np.ones((m, 1)), X.reshape(m, 1)))  # Добавление столбца единиц для свободного члена

# Инициализация параметров theta
theta = np.zeros(2)

# Вычисление начальной стоимости функции
cost = computeCost(X, y, theta)
print(f'Initial cost: {cost}')  # Вывод начальной стоимости

# Настройка параметров градиентного спуска
alpha = 0.01  # Скорость обучения
num_iters = 1500  # Количество итераций

# Выполнение градиентного спуска в зависимости от выбранного варианта
if option == '1':
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)
elif option == '2':
    theta, J_history = gradientDescentVectorized(X, y, theta, alpha, num_iters)
else:
    print("Неверный выбор. Завершение программы.")
    exit()

# Работа с обученной моделью
work(X, y, theta)

# Построение поверхности функции стоимости
theta0_vals = np.linspace(-10, 10, 100)  # 100 точек от -10 до 10 для theta0
theta1_vals = np.linspace(-10, 10, 100)  # 100 точек от -10 до 10 для theta1
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))  # Матрица значений функции стоимости

# Вычисление функции стоимости для каждой пары значений theta
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)  # Заполнение матрицы значениями функции стоимости

# Создание графика с использованием Plotly с прозрачностью
fig = go.Figure(data=[go.Surface(z=J_vals.T, x=theta0_vals, y=theta1_vals, opacity=0.7)])  # Установите opacity

# Добавление осей и заголовка
fig.update_layout(
    scene=dict(
        xaxis_title='Theta 0',
        yaxis_title='Theta 1',
        zaxis_title='Cost'
    ),
    title='Surface of Cost Function and Gradient Descent Path'
)

# Путь градиентного спуска
theta_history = []  # Список для хранения истории параметров
theta = np.zeros(2)  # Инициализация theta

# Выполнение градиентного спуска с сохранением истории
for _ in range(num_iters):
    theta = theta - (alpha / len(y)) * (X.T.dot(X.dot(theta) - y))
    theta_history.append(theta)  # Сохранение текущего значения theta

theta_history = np.array(theta_history)  # Преобразование в массив

# Вычисление значений стоимости для каждой итерации градиентного спуска
cost_history = np.array([computeCost(X, y, t) for t in theta_history])

# Добавление точек градиентного спуска на график
fig.add_trace(go.Scatter3d(
    x=theta_history[:, 0],
    y=theta_history[:, 1],
    z=cost_history,
    mode='markers',
    marker=dict(size=5, color='red')  # Настройка визуализации точек
))

# Отображение графика
fig.show()
