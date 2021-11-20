from pandas import DataFrame
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import util

def k_nn(x, data, k):
    def partition(start, end):
        def dist_sq(y):
            return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
        def swap(i, j):
            temp = data[i]
            data[i] = data[j]
            data[j] = temp
        pivot = random.randint(start, end-1)
        dp = dist_sq(data[pivot][0])
        swap(pivot, end-1)
        cutoff = start
        for i in range(start, end):
            di = dist_sq(data[i][0])
            if di < dp:
                swap(i, cutoff)
                cutoff += 1
        swap(cutoff, end-1)
        return cutoff
    def get_first_k(start, end):
        if start == end-1:
            return
        pivot = partition(start, end)
        if k-1 < pivot:
            get_first_k(start, pivot)
        elif k-1 > pivot:
            get_first_k(pivot+1, end)
    get_first_k(0, len(data))
    ctr = 0
    for i in range(k):
        ctr += data[i][1]
    return 1 if ctr > 0 else -1

def show_region(xdomain, ydomain, data, k):
    precision = 100
    for i in range(precision):
        for j in range(precision):
            x = [(xdomain[1] - xdomain[0]) / precision * i + xdomain[0], (ydomain[1] - ydomain[0]) / precision * j + ydomain[0]]
            value = k_nn(x, data, k)
            plt.plot(x[0], x[1], '.', color = 'red' if value == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    plus = mpatches.Patch(color = 'blue', label = '+1')
    plus = mpatches.Patch(color = 'red', label = '-1')
    plt.show()

def error(d_test, d_train, k):
    return sum([0 if k_nn(d[0], d_train, k_best) == d[1] else 1 for d in d_test]) / len(d_test)

if __name__ == '__main__':
    cutoff = 300
    (d_train, d_test) = util.get_data(cutoff)
    domain = (-1.1, 1.1)
    # show_region(domain, domain, d_train, 1)
    # show_region(domain, domain, d_train, 3)

    ks = [i for i in range(int(math.sqrt(cutoff) + 5)) if i % 2 == 1]
    e_cv = []

    k_best = 0
    e_best = 1
    for k in ks:
        e = 0
        for i in range(len(d_train)):
            d_minus = d_train[: i] + d_train[i+1 :]
            if k_nn(d_train[i][0], d_minus, k) != d_train[i][1]:
                e += 1
        e /= len(d_train)
        e_cv.append(e)
        if e <= e_best:
            e_best = e
            k_best = k
    print('best k:', k_best)
    print('cv error:', e_best)
    print('E_in:', error(d_train, d_train, k_best))
    print('E_test:', error(d_test, d_train, k_best))
    plt.plot(ks, e_cv, label = 'E_cv')
    plt.legend()
    plt.show()

    show_region(domain, domain, d_train, k_best)
