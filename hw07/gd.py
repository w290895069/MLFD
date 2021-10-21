from pandas import DataFrame
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def f(x, y):
    return x ** 2 + 2 * y ** 2 + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)

def grad_f(x, y):
    return (2*x + 4*math.pi*math.cos(2*math.pi*x)*math.sin(2*math.pi*y), 4*y + 4*math.pi*math.sin(2*math.pi*x)*math.cos(2*math.pi*y))

def gd(x, y, eta, num_itr):
    fields = {
        'Iteration': [],
        'f(x, y)': []
    }
    for i in range(num_itr):
        (dx, dy) = grad_f(x, y)
        x -= dx * eta
        y -= dy * eta
        fields['Iteration'].append(i)
        fields['f(x, y)'].append(f(x, y))
    df = DataFrame(fields, columns = ['Iteration', 'f(x, y)'])
    df.plot(x = 'Iteration', y = 'f(x, y)', kind = 'line')
    plt.show()
    return (x, y, f(x, y))

if __name__ == '__main__':
    # print(f(0.1, 0.1))
    print(gd(0.1, 0.1, 0.01, 50))
    # print(gd(0.1, 0.1, 0.1, 50))
    print(gd(1, 1, 0.01, 50))
    print(gd(-0.5, -0.5, 0.01, 50))
    print(gd(-1, -1, 0.01, 50))
