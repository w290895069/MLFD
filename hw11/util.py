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
    return ((get_bbox_area(pixels), get_curviness(pixels)), 1 if value == 1 else -1)

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

    limits0 = get_limits(0)
    limits1 = get_limits(1)
    return [([transform(d[0][0], limits0), transform(d[0][1], limits1)], d[1]) for d in data]


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

def get_data(cutoff):
    data = []
    for line in open('../ZipDigits.train.txt', 'r').readlines() + open('../ZipDigits.test.txt', 'r').readlines():
        data.append(parse(line))
    data = normalize(data)
    print(len(data) - cutoff)
    random.shuffle(data)
    return (data[: cutoff], data[cutoff :])
