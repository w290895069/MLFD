from pandas import DataFrame
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def show_region(xrange, yrange, data, k):
    precision = 100
    for i in range(precision):
        for j in range(precision):
            x = [(xrange[1] - xrange[0]) / precision * i + xrange[0], (yrange[1] - yrange[0]) / precision * j + yrange[0]]
            value = k_nn(x, data, k)
            plt.plot(x[0], x[1], '.', color = 'red' if value == -1 else 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plus = mpatches.Patch(color = 'blue', label = '+1')
    plus = mpatches.Patch(color = 'red', label = '-1')
    plt.show()

def show_region_z(xrange, yrange, data, k, transform):
    data = [(transform(d[0]), d[1]) for d in data]
    precision = 100
    for i in range(precision):
        for j in range(precision):
            x = [(xrange[1] - xrange[0]) / precision * i + xrange[0], (yrange[1] - yrange[0]) / precision * j + yrange[0]]
            value = k_nn(transform(x), data, k)
            plt.plot(x[0], x[1], '.', color = 'red' if value == -1 else 'blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plus = mpatches.Patch(color = 'blue', label = '+1')
    plus = mpatches.Patch(color = 'red', label = '-1')
    plt.show()

def improved_atan(x1, x2):
    if x1 == 0:
        return math.pi / 2 if x2 > 0 else -math.pi / 2 if x2 < 0 else 0
    return math.atan(x2 / x1)

if __name__ == '__main__':
    data = [([1,0], -1), ([0,1], -1), ([0,-1], -1), ([-1,0], -1), ([0,2], 1), ([0,-2], 1), ([-2,0], 1)]
    show_region((-3, 3), (-3, 3), data, 1)
    show_region((-3, 3), (-3, 3), data, 3)
    show_region_z((-3, 3), (-3, 3), data, 1, lambda x : [(x[0] ** 2 + x[1] ** 2) ** 0.5, improved_atan(x[0], x[1])])
    show_region_z((-3, 3), (-3, 3), data, 3, lambda x : [(x[0] ** 2 + x[1] ** 2) ** 0.5, improved_atan(x[0], x[1])])
