# Управляет работой обученной модели, включая вывод прогнозов
def work(X, y, theta):
    prediction = X.dot(theta)
    print(f'Predicted profits: {prediction}')
