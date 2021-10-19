import random, sys
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def dot(x, y):
    sum = 0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum

def slope(w):
    return -w[1] / w[2]

def intercept(w):
    return -w[0] / w[2]

def generate_data(count, thk, rad, sep):

    def classify(x, y):
        if abs(y) < sep / 2:
            return 0
        sign = 1 if y > 0 else -1
        y -= sign * (sep / 2)
        x += sign * (rad + thk / 2) / 2
        r2 = x ** 2 + y ** 2
        if r2 < rad ** 2 or r2 > (rad + thk) ** 2:
            return 0
        return -sign

    data = []
    x_span = rad * 3 + thk * 5 / 2
    y_span = (rad + thk) * 2 + sep
    ctr = 0
    while ctr < count:
        x = random.uniform(-x_span / 2, x_span / 2)
        y = random.uniform(-y_span / 2, y_span / 2)
        val = classify(x, y)
        if val != 0:
            data.append(((1, x, y), val))
            ctr += 1
    return data

def perceptron(data):
    w = [0, 0, 0]
    while True:
        ctr = 0
        for datum in data:
            product = dot(datum[0], w)
            if product * datum[1] <= 0:
                for i in range(len(w)):
                    w[i] += datum[0][i] * datum[1]
                ctr += 1
        if ctr == 0:
            break
    return w

def regression(data):
    y = np.array([d[1] for d in data])
    x = np.array([d[0] for d in data])
    xt = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(xt, x)), np.matmul(xt, y))

if __name__ == '__main__':
    data = generate_data(2000, 5, 10, 5)
    x = np.linspace(-22, 22, 2)

    fields = {
        'x': [datum[0][1] for datum in data],
        'y': [datum[0][2] for datum in data]
    }
    df = DataFrame(fields, columns = ['x', 'y'])
    df.plot(x = 'x', y = 'y', kind = 'scatter', color = ['red' if datum[1] < 0 else 'blue' for datum in data], figsize = (10, 10))

    plus = mpatches.Patch(color='blue', label='f(x, y) = +1')
    minus = mpatches.Patch(color='red', label='f(x, y) = -1')

    w = perceptron(data)
    # w = regression(data)
    y_g = slope(w) * x + intercept(w)
    plt.plot(x, y_g, color = 'green')
    g = mpatches.Patch(color='green', label='g(x, y) = {:.2f} + {:.2f}x + {:.2f}y = 0'.format(w[0], w[1], w[2]))
    plt.legend(handles=[plus, minus, g])
    plt.show()
