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
    t = 0
    while True:
        ctr = 0
        for datum in data:
            product = dot(datum[0], w)
            if product * datum[1] <= 0:
                for i in range(len(w)):
                    w[i] += datum[0][i] * datum[1]
                ctr += 1
                t += 1
        if ctr == 0:
            break
    return (w, t)

if __name__ == '__main__':
    fields = {
        'sep': [],
        'iterations': []
    }
    for i in range(1, 26):
        sep = i / 5
        data = generate_data(2000, 5, 10, sep)
        fields['sep'].append(sep)
        fields['iterations'].append(perceptron(data)[1])

    print(fields)
    df = DataFrame(fields, columns = ['sep', 'iterations'])
    df.plot(x = 'sep', y = 'iterations', kind = 'line')
    plt.show()
