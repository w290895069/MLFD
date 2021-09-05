import random, sys
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

w_f = [1, 2, 3]

def dot(x, y):
    sum = 0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum

def f(x):
    val = dot(x, w_f)
    return 1 if val > 0 else -1 if val < 0 else 0

def slope(w):
    return -w[1] / w[2]

def intercept(w):
    return -w[0] / w[2]

def generate_data(count):
    data = []
    for i in range(count):
        x = [1, random.uniform(-10, 10), random.uniform(-10, 10)]
        y = f(x)
        if y == 0:
            continue
        data.append((x, y))
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

if __name__ == '__main__':
    data = generate_data(20)
    x = np.linspace(-10, 10, 2)
    y_f = slope(w_f) * x + intercept(w_f)

    fields = {
        'x1': [datum[0][1] for datum in data],
        'x2': [datum[0][2] for datum in data]
    }
    df = DataFrame(fields, columns = ['x1', 'x2'])
    df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
    plt.plot(x, y_f)

    plus = mpatches.Patch(color='green', label='f(x) = +1')
    minus = mpatches.Patch(color='red', label='f(x) = -1')
    zero = mpatches.Patch(color='blue', label='f(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w_f[0], w_f[1], w_f[2]))
    plt.legend(handles=[plus, minus, zero])
    plt.show()

    w = perceptron(data)
    y_g = slope(w) * x + intercept(w)
    df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
    plt.plot(x, y_f)
    plt.plot(x, y_g, color = 'cyan')
    g = mpatches.Patch(color='cyan', label='g(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w[0], w[1], w[2]))
    plt.legend(handles=[plus, minus, zero, g])
    plt.show()

    for count in [20, 100, 1000]:
        data = generate_data(count)
        fields = {
            'x1': [datum[0][1] for datum in data],
            'x2': [datum[0][2] for datum in data]
        }
        df = DataFrame(fields, columns = ['x1', 'x2'])
        w = perceptron(data)
        y_g = slope(w) * x + intercept(w)
        df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
        plt.plot(x, y_f)
        plt.plot(x, y_g, color = 'cyan')
        g = mpatches.Patch(color='cyan', label='g(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w[0], w[1], w[2]))
        plt.legend(handles=[plus, minus, zero, g])
        plt.show()
