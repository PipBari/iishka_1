import numpy as np
import plotly.graph_objects as go
from warmUpExercise import warmUpExercise
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from work import work

# Загрузка данных
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]  # Количество автомобилей
y = data[:, 1]  # Прибыль СТО

# Визуализация данных
plotData(X, y)

# Настройка параметров
m = len(y)  # Количество обучающих примеров
X = np.hstack((np.ones((m, 1)), X.reshape(m, 1)))  # Добавление столбца единиц

# Инициализация theta
theta = np.zeros(2)

# Вычисление стоимости
cost = computeCost(X, y, theta)
print(f'Initial cost: {cost}')

# Градиентный спуск
alpha = 0.01
num_iters = 1500
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Работа с обученной системой
work(X, y, theta)

# Построение поверхности функции стоимости
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i, j] = computeCost(X, y, t)

# Создание графика с использованием Plotly
fig = go.Figure(data=[go.Surface(z=J_vals.T, x=theta0_vals, y=theta1_vals)])

# Добавление осей и заголовка
fig.update_layout(
    scene=dict(
        xaxis_title='Theta 0',
        yaxis_title='Theta 1',
        zaxis_title='Cost'
    ),
    title='Surface of Cost Function and Gradient Descent Path'
)

# Отображение графика
# Путь градиентного спуска
theta_history = []
theta = np.zeros(2)

for _ in range(num_iters):
    theta = theta - (alpha / len(y)) * (X.T.dot(X.dot(theta) - y))
    theta_history.append(theta)

theta_history = np.array(theta_history)

# Вычисляем значения стоимости для каждого шага градиентного спуска
cost_history = np.array([computeCost(X, y, t) for t in theta_history])

# Добавление точек градиентного спуска на график
fig.add_trace(go.Scatter3d(
    x=theta_history[:, 0],
    y=theta_history[:, 1],
    z=cost_history,
    mode='markers',
    marker=dict(size=5, color='red')
))

# Отображение графика
fig.show()
