from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

def parse(line):
    fields = line.split(' ')
    assert len(fields) == 258
    value = int(float(fields[0]))
    grid = [[float(fields[i * 16 + j + 1]) > 0 for j in range(16)] for i in range(16)]
    pixels = set([(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if grid[i][j]])
    return (pixels, value)

def visualize(pixels):
    fields = {
        'x': [p[1] for p in pixels],
        'y': [15 - p[0] for p in pixels]
    }
    df = DataFrame(fields, columns = ['x', 'y'])
    df.plot(x = 'x', y = 'y', kind = 'scatter', xlim = (-1, 16), ylim = (-1, 16), figsize = (8, 8), s = 100)
    plt.show()

def visualize_2():
    one = None
    five = None
    for line in open('ZipDigits.train.txt', 'r').readlines():
        datum = parse(line)
        if datum[1] == 1:
            one = datum[0]
        if datum[1] == 5:
            five = datum[0]
        if one != None and five != None:
            break
    visualize(one)
    visualize(five)

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


def visualize_features():
    fields = {
        'Bounding Box Area': [],
        'Curviness': []
    }
    for line in open('ZipDigits.train.txt', 'r').readlines() + open('ZipDigits.test.txt', 'r').readlines():
        datum = parse(line)
        if datum[1] != 1 and datum[1] != 5:
            continue
        plt.plot(get_bbox_area(datum[0]), get_curviness(datum[0]), 'x' if datum[1] == 5 else 'o', color = 'red' if datum[1] == 5 else 'blue')
        fields['Bounding Box Area'].append(get_bbox_area(datum[0]))
        fields['Curviness'].append(get_curviness(datum[0]))
    plt.xlabel('Bounding Box Area')
    plt.ylabel('Curviness')
    plt.show()


if __name__ == '__main__':
    visualize_features()
    visualize_2()
