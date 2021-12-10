from pandas import DataFrame
import numpy as np
from cvxopt import matrix, solvers
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from digits import get_bbox_area, get_curviness

def parse(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    return (np.array([get_bbox_area(pixels), get_curviness(pixels)]), 1 if value == 1 else -1)

def make_preprocess(data):
    m_x = np.array([v_x for (v_x, y) in data])
    v_xmean = np.mean(m_x, axis = 0)
    v_xstd = np.std(m_x, axis = 0)
    return lambda d : [(np.divide(v_x - v_xmean, v_xstd), float(y)) for (v_x, y) in d]

def q3c():
    xrange = np.arange(-5, 5, 0.1)
    yrange = np.arange(-5, 5, 0.1)
    x, y = np.meshgrid(xrange, yrange)
    eq_x = x
    eq_z = x ** 3 - y
    plt.contour(x, y, eq_x, [0], colors = ['blue'])
    plt.contour(x, y, eq_z, [0], colors = ['green'])
    plt.plot([], [], color = 'blue', label = 'X Space')
    plt.plot([], [], color = 'green', label = 'Z Space')
    plt.scatter([-1, 1], [0, 0], color = 'black')
    plt.legend()
    plt.show()

def kernel(v_x1, v_x2):
    return (v_x1[0] * v_x2[0] + v_x1[1] * v_x2[1] + 1) ** 8
    # return np.dot(v_x1, v_x2)

def train(data, c):
    m_p = matrix([[yn * ym * kernel(v_xn, v_xm) for (v_xn, yn) in data] for (v_xm, ym) in data])
    v_q = matrix([-1.0 for d in data])
    m_g = matrix([[1.0 if j == i else 0.0 for j in range(len(data))] + [-1.0 if j == i else 0.0 for j in range(len(data))] for i in range(len(data))])
    v_h = matrix([c for d in data] + [0.0 for d in data])
    m_a = matrix([[y] for (v_x, y) in data])
    v_b = matrix([0.0])
    v_alpha = solvers.qp(m_p, v_q, m_g, v_h, m_a, v_b)['x']
    b = 0
    v_xs = None
    ys = None
    cutoff = min(c / 1000, 10 ** (-7))
    for i in range(len(data)):
        if v_alpha[i] > cutoff:
            (v_xs, ys) = data[i]
            break
    b = ys
    v_alphas = []
    m_xs = []
    v_ys = []
    for i in range(len(data)):
        alpha = v_alpha[i]
        # if alpha >= cutoff:
        (v_x, y) = data[i]
        b -= v_alpha[i] * y * kernel(v_x, v_xs)
        m_xs.append(v_x)
        v_ys.append(y)
        v_alphas.append(alpha)
    return (m_xs, np.array(v_ys), np.array(v_alphas), b)
    # print(ctr)

def compute(classifier, v_x):
    (m_xs, v_ys, v_alphas, b) = classifier
    sum = b
    # print(v_x)
    for i in range(len(v_alphas)):
        sum += v_alphas[i] * v_ys[i] * kernel(m_xs[i], v_x)
    return sum

def classify(classifier, v_x):
    return 1 if compute(classifier, v_x) > 0 else -1

def plot_boundary(classifier, data):
    for (v_x, y) in data:
        plt.plot(v_x[0], v_x[1], 'x' if y == -1 else 'o', color = 'red' if y == -1 else 'blue')
    xrange = np.arange(-3, 3, 0.1)
    yrange = np.arange(-3, 3, 0.1)
    x, y = np.meshgrid(xrange, yrange)
    eq = compute(classifier, np.array([x, y], dtype=object))
    plt.contour(x, y, eq, [0], linewidths=[0.5])
    precision = 50
    for i in range(precision):
        for j in range(precision):
            x = 6.0 / precision * i - 3
            y = 6.0 / precision * j - 3
            plt.plot(x, y, 's', color = 'blue' if classify(classifier, np.array([x, y], dtype=object)) == 1 else 'red', alpha = 0.1)
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    plt.show()

def error(classifier, data):
    sum = 0
    for (v_x, y) in data:
        if classify(classifier, v_x) != y:
            sum += 1
    return sum / len(data)

if __name__ == '__main__':
    # q3c()

    data = [parse(line) for line in open('../ZipDigits.train.txt', 'r').readlines() + open('../ZipDigits.test.txt', 'r').readlines()]
    random.shuffle(data)
    cutoff = 300

    d_train = data[: cutoff]
    d_test = data[cutoff :]
    preprocess = make_preprocess(d_train)
    d_train = preprocess(d_train)
    d_test = preprocess(d_test)

    # classifier = train(d_train, 0.0001)
    # print(error(classifier, d_train))
    # plot_boundary(classifier, d_train)

    c_candidates = [2 ** x for x in range(-10, 11)]
    e_opt = 1
    c_opt = 0
    for c in c_candidates:
        e = 0
        for i in range(len(d_train)):
            d_minus = d_train[: i] + d_train[i+1 :]
            d_val = [d_train[i]]
            classifier = train(d_train, c)
            e += error(classifier, d_train)
        e /= len(d_train)
        if e < e_opt:
            e_opt = e
            c_opt = c
    print('C_opt:', c_opt)
    classifier_opt = train(d_train, c_opt)
    print('E_test:', error(classifier_opt, d_test))
    plot_boundary(classifier_opt, d_train)
