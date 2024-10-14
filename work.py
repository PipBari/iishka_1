import numpy as np

def work(X, y, theta):
    prediction = X.dot(theta)
    print(f'Predicted profits: {prediction}')
