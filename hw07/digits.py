from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    return (np.array([1, get_bbox_area(pixels), get_curviness(pixels)]), 1 if value == 1 else -1 if value == 5 else 0)

def parse_3o(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    x = get_bbox_area(pixels)
    y = get_curviness(pixels)
    return (np.array([1, x, y, x ** 2, x * y, y ** 2, x ** 3, x ** 2 * y, x * y ** 2, y ** 3]), 1 if value == 1 else -1 if value == 5 else 0)

def slope(w):
    return -w[1] / w[2]

def intercept(w):
    return -w[0] / w[2]

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


def visualize_features(train, test):
    for d in train:
        plt.plot(d[0][1], d[0][2], 'x' if d[1] == -1 else 'o', color = 'red' if d[1] == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    w = pocket(train, regression(train))
    print(error(train, w))
    x = np.linspace(0, 240, 2)
    y = slope(w) * x + intercept(w)
    plt.plot(x, y, color = 'green')
    plt.show()
    for d in test:
        plt.plot(d[0][1], d[0][2], 'x' if d[1] == -1 else 'o', color = 'red' if d[1] == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    print(error(test, w))
    plt.plot(x, y, color = 'green')
    plt.show()

def visualize_features_3o(train, test):
    for d in train:
        plt.plot(d[0][1], d[0][2], 'x' if d[1] == -1 else 'o', color = 'red' if d[1] == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    w = pocket(train, regression(train))
    print(error(train, w))
    xrange = np.arange(0, 240, 1)
    yrange = np.arange(0, 70, 1)
    x, y = np.meshgrid(xrange, yrange)
    eq = w[0] + w[1]*x + w[2]*y + w[3]*x**2 + w[4]*x*y + w[5]*y**2 + w[6]*x**3 + w[7]*x**2*y + w[8]*x*y**2 + w[9]*y**3
    plt.contour(x, y, eq, [0])
    plt.show()

    for d in test:
        plt.plot(d[0][1], d[0][2], 'x' if d[1] == -1 else 'o', color = 'red' if d[1] == -1 else 'blue')
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    print(error(test, w))
    plt.contour(x, y, eq, [0])
    plt.show()

def regression(data):
    y = np.array([d[1] for d in data])
    x = np.array([d[0] for d in data])
    xt = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(xt, x)), np.matmul(xt, y))

def error(data, w):
    ctr = 0
    for d in data:
        if np.dot(w, d[0]) * d[1] <= 0:
            ctr += 1
    return ctr / len(data)


def pocket(data, w):
    updates = 0
    w_hat = w
    old_ein = error(data, w)
    while True:
        for d in data:
            if np.dot(d[0], w) * d[1] <= 0:
                w = np.array([w[i] + d[0][i] * d[1] for i in range(len(w))])
                new_ein = error(data, w)
                if new_ein < old_ein:
                    w_hat = w
                    old_ein = new_ein
                    print(old_ein)
                updates += 1
                if updates == 1000:
                    return w_hat
    return w_hat

if __name__ == '__main__':
    # train = [parse(line) for line in open('ZipDigits.train.txt', 'r').readlines()]
    # train = [d for d in train if d[1] == 1 or d[1] == -1]
    # test = [parse(line) for line in open('ZipDigits.test.txt', 'r').readlines()]
    # test = [d for d in test if d[1] == 1 or d[1] == -1]
    # print(len(train), len(test))
    # visualize_features(train, test)

    train = [parse_3o(line) for line in open('ZipDigits.train.txt', 'r').readlines()]
    train = [d for d in train if d[1] == 1 or d[1] == -1]
    test = [parse_3o(line) for line in open('ZipDigits.test.txt', 'r').readlines()]
    test = [d for d in test if d[1] == 1 or d[1] == -1]
    print(len(train), len(test))
    visualize_features_3o(train, test)
