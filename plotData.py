import matplotlib.pyplot as plt

def plotData(X, y):
    plt.scatter(X, y, color='red', marker='x', label='Training data')
    plt.xlabel('Number of Cars')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()
