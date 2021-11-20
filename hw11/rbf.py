from pandas import DataFrame
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util

def dist(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

def gaussian_kernel(x, mu, k):
    return math.e ** ((-(x[0] - mu[0]) ** 2 + (x[1] - mu[1]) ** 2) * k / 8)

def k_center(data, k):
    centers = [random.choice(data)]
    for i in range(k-1):
        max_dist = 0
        max_point = None
        for x in data:
            x_dist = min([dist(x, c) for c in centers])
            if x_dist > max_dist:
                max_dist = x_dist
                max_point = x
        centers.append(max_point)
    return centers

def transform(x, centers):
    return [gaussian_kernel(x, c, len(centers)) for c in centers]

def regression(x, y):
    x = np.array(x)
    y = np.array(y)
    xt = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(xt, x)), np.matmul(xt, y))

def rbf(x, w, centers):
    gk = [1] + [gaussian_kernel(x, c, len(centers)) for c in centers]
    return 1 if np.dot(np.array(w), np.array(gk)) > 0 else -1

def show_region(xdomain, ydomain, w, centers):
    precision = 100
    for i in range(precision):
        for j in range(precision):
            x = [(xdomain[1] - xdomain[0]) / precision * i + xdomain[0], (ydomain[1] - ydomain[0]) / precision * j + ydomain[0]]
            value = rbf(x, w, centers)
            plt.plot(x[0], x[1], '.', color = 'red' if value == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    plus = mpatches.Patch(color = 'blue', label = '+1')
    plus = mpatches.Patch(color = 'red', label = '-1')
    plt.show()

def error(d_test, w, centers):
    return sum([0 if rbf(d[0], w, centers) == d[1] else 1 for d in d_test]) / len(d_test)

if __name__ == '__main__':
    cutoff = 300
    (d_train, d_test) = util.get_data(cutoff)
    k = 20

    ks = range(1, 20)
    e_cv = []

    k_best = 0
    e_best = 1
    c_best = None
    w_best = None
    for k in ks:
        e = 0
        for i in range(len(d_train)):
            d_minus = d_train[: i] + d_train[i+1 :]
            centers = k_center([d[0] for d in d_minus], k)
            z = [[1] + transform(d[0], centers) for d in d_minus]
            y = [d[1] for d in d_minus]
            w = regression(z, y)
            if rbf(d_train[i][0], w, centers) != d_train[i][1]:
                e += 1
        e /= len(d_train)
        e_cv.append(e)
        if e <= e_best:
            e_best = e
            k_best = k
            w_best = w
            c_best = centers
    print('best k:', k_best)
    print('cv error:', e_best)
    print('E_in:', error(d_train, w_best, c_best))
    print('E_test:', error(d_test, w_best, c_best))
    plt.plot(ks, e_cv, label = 'E_cv')
    plt.legend()
    plt.show()

    domain = (-1.1, 1.1)
    show_region(domain, domain, w, centers)
