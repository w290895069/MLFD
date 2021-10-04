import random, sys
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def get_line(datum):
    return (datum[0] + datum[1], -datum[0]*datum[1])

def eval(g, x):
    return g[0] * x + g[1]

if __name__ == '__main__':
    data = [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(1000)]
    test = [random.uniform(-1, 1) for i in range(1000)]

    g_l = [get_line(d) for d in data]

    g_bar = (sum([g[0] for g in g_l]) / len(g_l), sum([g[1] for g in g_l]) / len(g_l))

    var = sum([(eval(g, x) - eval(g_bar, x)) ** 2 for x in test for g in g_l]) / (len(g_l) * len(test))

    bias = sum([(x*x - eval(g_bar, x)) ** 2 for x in test]) / len(test)

    e_out = var + bias

    print(g_bar)
    print(var)
    print(bias)
    print(e_out)

    x = np.linspace(-1, 1, 100)
    f = x * x
    plt.plot(x, f, color = 'blue')
    g = eval(g_bar, x)
    plt.plot(x, g, color = 'red')

    #
    pf = mpatches.Patch(color='blue', label='f(x) = x^2')
    pg = mpatches.Patch(color='red', label='g_bar(x) = {:.4f}x + {:.4f}'.format(g_bar[0], g_bar[1]))
    plt.legend(handles=[pf, pg])
    plt.show()
