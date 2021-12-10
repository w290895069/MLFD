from pandas import DataFrame
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

import nn

def parse(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    return (np.array([1, get_bbox_area(pixels), get_curviness(pixels)]), 1 if value == 1 else -1)

def make_preprocess(data):
    m_x = np.array([v_x for (v_x, y) in data])
    v_xmean = np.mean(m_x, axis = 0)
    v_xmean[0] = 0.0
    v_xstd = np.std(m_x, axis = 0)
    v_xstd[0] = 1.0
    return lambda d : [(np.divide(v_x - v_xmean, v_xstd), float(y)) for (v_x, y) in d]

def get_bbox_area(pixels):
    x_min = 15
    x_max = 0
    y_min = 15
    y_max = 0
    for p in pixels:
        x_min = min(x_min, p[0])
        x_max = max(x_max, p[0])
        y_min = min(y_min, p[1])
        y_max = max(y_max, p[1])
    return (x_max - x_min) * (y_max - y_min)

def get_curviness(pixels):
    neighbors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    indeces = {neighbors[i] : i for i in range(len(neighbors))}
    def get_start_next(p):
        found = False
        angle = 0
        for i in range(len(neighbors) * 2):
            dr = neighbors[i % len(neighbors)]
            next = (p[0] + dr[0], p[1] + dr[1])
            if next in pixels:
                found = True
            if found:
                if next not in pixels:
                    angle += 1
                if angle > 0 and next in pixels:
                    return (next, angle + 1)
        return None

    def get_next(p, dr):
        angle = 0
        offset = indeces[dr]
        for i in range(1, len(neighbors) + 1):
            dr = neighbors[(i + offset) % len(neighbors)]
            next = (p[0] + dr[0], p[1] + dr[1])
            if next in pixels:
                return (next, angle + 1)
            angle += 1
        return None

    start = None
    result = None
    curr = None
    angle = 0
    for i in range(16):
        for j in range(16):
            if (i, j) in pixels:
                result = get_start_next((i, j))
                if result != None:
                    start = (i, j)
                    (curr, angle) = result
                    break
        if start != None:
            break

    prev = start
    sum = abs(angle - 4)
    next = None
    ctr = 1
    while next != start:
        (next, angle) = get_next(curr, (prev[0] - curr[0], prev[1] - curr[1]))
        sum += abs(angle - 4)
        ctr += 1
        prev = curr
        curr = next
    return sum

def error(data, w, theta):
    sum = 0
    for (x, y) in data:
        sum += (nn.forward(x, w, theta)[3] - y) ** 2
    return sum / len(data) / 4

def plot_boundary(data, w):
    for (x, y) in data:
        plt.plot(x[1], x[2], 'x' if y == -1 else 'o', color = 'red' if y == -1 else 'blue')
    xrange = np.arange(-3, 3, 0.1)
    yrange = np.arange(-3, 3, 0.1)
    x, y = np.meshgrid(xrange, yrange)
    eq = nn.forward(np.array([1, x, y], dtype=object), w, identity, object)[3]
    plt.contour(x, y, eq, [0], linewidths=[0.5])
    precision = 50
    for i in range(precision):
        for j in range(precision):
            x = 6.0 / precision * i - 3
            y = 6.0 / precision * j - 3
            plt.plot(x, y, 's', color = 'blue' if nn.classify(w, np.array([1.0, x, y])) == 1 else 'red', alpha = 0.1)
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    plt.show()

def identity(x):
    return x

def identity_p(x):
    return 1


def sign(x):
    return 1 if x > 0 else -1

def train(data, lm, num_itr):
    w = nn.init_nn(10, lambda: random.uniform(-1, 1))
    # print(w)
    # plot_boundary(data, w)
    eta = 0.1
    iterations = []
    eins = []
    ein = error(data, w, identity)
    gw = nn.get_gradients(data, w, identity, identity_p)
    updated = False
    for i in range(num_itr):
        if updated:
            gw = nn.get_gradients(data, w, identity, identity_p)

        diff_w = nn.update(w, w, 2 * lm * eta)
        nn.update(w, gw, eta)
        new_ein = error(data, w, identity)
        if new_ein < ein:
            ein = new_ein
            eta *= 1.01
            updated = True
        else:
            nn.update(w, gw, -eta)
            nn.update(w, diff_w, -1)
            eta *= 0.4
            updated = False
        if i % (num_itr // 100) == 0:
            # print(str(int(i / num_itr * 100)) + '% with ' + str(num_itr) + ' iterations')
            eins.append(ein)
            iterations.append(i+1)
    return (w, iterations, eins)

def plot_iterations(iterations, eins):
    plt.plot(iterations, eins)
    plt.xlabel('iteration')
    plt.ylabel('E_in')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    data = [parse(line) for line in open('../ZipDigits.train.txt', 'r').readlines() + open('../ZipDigits.test.txt', 'r').readlines()]
    cutoff = 300
    # print(len(data) - cutoff)
    random.shuffle(data)
    d_train = data[: cutoff]
    d_test = data[cutoff :]

    preprocess = make_preprocess(d_train)
    d_train = preprocess(d_train)
    d_test = preprocess(d_test)

    # (w, iterations, eins) = train(d_train, 0, 2000000)
    # print('E_in:', error(d_train, w, sign))
    # print(w)
    # plot_iterations(iterations, eins)
    # plot_boundary(d_train, w)

    # (w, iterations, eins) = train(d_train, 0.01 / len(d_train), 2000000)
    # print('E_in:', error(d_train, w, sign))
    # print(w)
    # plot_iterations(iterations, eins)
    # plot_boundary(d_train, w)

    cutoff = 250
    d_val = d_train[cutoff: ]
    d_train = d_train[: cutoff]
    num_itr_best = 0
    e_best = 9999999

    for num_itr in [100 * i for i in range(1, 21)]:
        w = train(d_train, 0, num_itr)[0]
        e_val = error(d_val, w, sign)
        if e_val < e_best:
            e_best = e_val
            num_itr_best = num_itr

    d_train += d_val
    w = train(d_train, 0, num_itr_best)[0]
    print('best iterations:', num_itr_best)
    print('best E_val:', e_best)
    print('E_test:', error(d_test, w, sign))
    plot_boundary(d_train, w)
