from pandas import DataFrame
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    return ((1, get_bbox_area(pixels), get_curviness(pixels)), 1 if value == 1 else -1)

def normalize(data):
    def get_limits(index):
        lo = 999999
        hi = 0
        for d in data:
            value = d[0][index]
            if value > hi:
                hi = value
            if value < lo:
                lo = value
        return (lo, hi)

    def transform(value, limits):
        return 2.0 * (value - limits[0]) / (limits[1] - limits[0]) - 1

    limits1 = get_limits(1)
    limits2 = get_limits(2)
    return [([1, transform(d[0][1], limits1), transform(d[0][2], limits2)], d[1]) for d in data]

def transform(features, order):
    def legendre(x):
        l = [1]
        if order == 0:
            return l
        l.append(x)
        for i in range(2, order+1):
            l.append((2*i-1) / float(i) * x * l[i-1] - (i-1) / float(i) * l[i-2])
        return l
    x = features[1]
    y = features[2]
    lx = legendre(x)
    ly = legendre(y)
    return [lx[i-j] * ly[j] for i in range(order+1) for j in range(i+1)]


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

def regression(data, l):
    y = np.array([d[1] for d in data])
    x = np.array([d[0] for d in data])
    xt = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(xt, x) + l * np.identity(len(xt))), np.matmul(xt, y))

def error(data, w):
    ctr = 0
    for d in data:
        if np.dot(w, d[0]) * d[1] <= 0:
            ctr += 1
    return ctr / len(data)

def plot_boundary(data, w):
    for d in data:
        plt.plot(d[0][1], d[0][2], 'x' if d[1] == -1 else 'o', color = 'red' if d[1] == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    xrange = np.arange(-1.1, 1.1, 0.001)
    yrange = np.arange(-1.1, 1.1, 0.001)
    x, y = np.meshgrid(xrange, yrange)
    eq = np.dot(w, np.array(transform((1, x, y), order), dtype=object))
    plt.contour(x, y, eq, [0])
    plt.show()

if __name__ == '__main__':
    data = []
    for line in open('ZipDigits.train.txt', 'r').readlines() + open('ZipDigits.test.txt', 'r').readlines():
        data.append(parse(line))
    data = normalize(data)
    order = 8
    d_transformed = [(transform(d[0], order), d[1]) for d in data]
    cutoff = 300
    print(len(data) - cutoff)
    random.shuffle(d_transformed)
    d_train = d_transformed[: cutoff]
    d_test = d_transformed[cutoff :]

    w = regression(d_train, 0)
    plot_boundary(d_train, w)

    w = regression(d_train, 2)
    plot_boundary(d_train, w)


    lm = [i / 100.0 for i in range(301)]
    e_cv = []
    e_test = []

    w_best = None
    e_best = 1
    l_best = 0
    for l in lm:
        e = 0
        for i in range(len(d_train)):
            d_cv = [d_train[i]]
            d_minus = d_train[: i] + d_train[i+1 :]
            w_minus = regression(d_minus, l)
            e += error(d_cv, w_minus)
        e /= len(d_train)
        e_cv.append(e)
        w_reg = regression(d_train, l)
        e_test.append(error(d_test, w_reg))
        if e <= e_best:
            e_best = e
            w_best = w_reg
            l_best = l
    plt.plot(lm, e_cv, label = 'E_cv')
    plt.plot(lm, e_test, label = 'E_test')
    plt.legend()
    plt.show()

    print(l_best, error(d_test, w_best))
    plot_boundary(d_test, w_best)
