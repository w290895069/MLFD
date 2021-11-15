import random
from p6_1 import k_nn, show_region

def generate_data(count, thk, rad, sep):

    def classify(x, y):
        if abs(y) < sep / 2:
            return 0
        sign = 1 if y > 0 else -1
        y -= sign * (sep / 2)
        x += sign * (rad + thk / 2) / 2
        r2 = x ** 2 + y ** 2
        if r2 < rad ** 2 or r2 > (rad + thk) ** 2:
            return 0
        return -sign

    data = []
    x_span = rad * 3 + thk * 5 / 2
    y_span = (rad + thk) * 2 + sep
    ctr = 0
    while ctr < count:
        x = random.uniform(-x_span / 2, x_span / 2)
        y = random.uniform(-y_span / 2, y_span / 2)
        val = classify(x, y)
        if val != 0:
            data.append(((x, y), val))
            ctr += 1
    return data

if __name__ == '__main__':
    data = generate_data(2000, 5, 10, 5)
    show_region((-22, 22), (-15, 15), data, 1)
    show_region((-22, 22), (-15, 15), data, 3)
