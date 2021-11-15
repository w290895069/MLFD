import random
import time

def generate_uniform(n):
    return [[random.uniform(0, 1), random.uniform(0, 1)] for i in range(n)]

def generate_gaussian(n):
    bumps = generate_uniform(10)
    data = []
    for i in range(n):
        bump = random.choice(bumps)
        data.append([random.gauss(bump[0], 0.1), random.gauss(bump[1], 0.1)])
    return data

def dist(x, y):
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5


class BVH:
    def __init__(self, data, center):
        self.center = center
        self.radius = max([dist(x, center) for x in data])
        if len(data) <= 10:
            self.points = data
            self.subtrees = None
        else:
            self.points = None
            new_centers = [random.choice(data)]
            for i in range(9):
                max_dist = 0
                max_point = None
                for x in data:
                    x_dist = min([dist(x, c) for c in new_centers])
                    if x_dist > max_dist:
                        max_dist = x_dist
                        max_point = x
                new_centers.append(max_point)
            new_data = [[] for i in range(len(new_centers))]
            for x in data:
                min_dist = 999999
                min_index = -1
                for i in range(len(new_centers)):
                    dist_i = dist(x, new_centers[i])
                    if dist_i < min_dist:
                        min_dist = dist_i
                        min_index = i
                new_data[min_index].append(x)
            self.subtrees = [BVH(new_data[i], new_centers[i]) for i in range(len(new_centers))]

    def get_closest(self, x):
        if self.subtrees == None:
            min_dist = 999999
            min_point = None
            for p in self.points:
                dist_x = dist(x, p)
                if dist_x < min_dist:
                    min_dist = dist_x
                    min_point = p
            return (min_point, min_dist)
        (min_point, min_dist) = self.subtrees[0].get_closest(x)
        for i in range(1, 10):
            if dist(x, self.subtrees[i].center) - self.subtrees[i].radius >= min_dist:
                continue
            (new_point, new_dist) = self.subtrees[i].get_closest(x)
            if new_dist < min_dist:
                min_dist = new_dist
                min_point = new_point
        return (min_point, min_dist)

def brute_force(x, data):
    min_dist = 999999
    min_point = None
    for p in data:
        d = dist(x, p)
        if d < min_dist:
            min_dist = d
            min_point = p
    return (min_point, min_dist)

def time_ms():
    return int(time.time() * 1000)

def experiment(generate_function):
    num_data = 10000
    data = generate_function(num_data)
    bvh = BVH(data, random.choice(data))
    num_itr = 10000

    t = time_ms()
    for i in range(num_itr):
        bvh.get_closest([random.uniform(-2, 2), random.uniform(-2, 2)])
    print(time_ms() - t)

    t = time_ms()
    for i in range(num_itr):
        brute_force([random.uniform(-2, 2), random.uniform(-2, 2)], data)
    print(time_ms() - t)

def close_enough(x, y):
    return dist(x, y) < 0.00001

if __name__ == '__main__':
    experiment(generate_uniform)
    experiment(generate_gaussian)
