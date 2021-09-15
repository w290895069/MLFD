import random, sys
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

def experiment():
    num_heads = [0] * 1000
    for i in range(1000):
        for j in range(10):
            if random.random() < 0.5:
                num_heads[i] += 1
    # print(num_heads)
    return (num_heads[0], random.choice(num_heads), min(num_heads))

def plot_epsilon(freq, rep):
    data = []
    for i in range(6):
        epsilon = i / 10
        # print(epsilon)
        sum = 0
        for j in range(6 - i - 1):
            sum += freq[j]
        for j in range(6 + i + 1, 11):
            sum += freq[j]
        # print(sum / rep, 2 * pow(math.e, -2 * epsilon * epsilon * rep))
        data.append(sum / rep)
    x = np.linspace(0, 0.5, 100)
    y = 2 * pow(math.e, -2 * x * x * 10)
    fields = {
        'epsilon': [i / 10 for i in range(6)],
        'probability': data
    }
    df = DataFrame(fields, columns = ['epsilon', 'probability'])
    df.plot(x = 'epsilon', y = 'probability', kind = 'line', color = 'blue')
    plt.plot(x, y, color = 'red')
    default = mpatches.Patch(color='red', label='Hoeffding bound')
    actual = mpatches.Patch(color='blue', label='Probability')
    plt.legend(handles=[default, actual])
    plt.show()

if __name__ == '__main__':
    # data = generate_data(20)
    # x = np.linspace(-10, 10, 2)
    # y_f = slope(w_f) * x + intercept(w_f)
    #
    # fields = {
    #     'x1': [datum[0][1] for datum in data],
    #     'x2': [datum[0][2] for datum in data]
    # }
    # df = DataFrame(fields, columns = ['x1', 'x2'])
    # df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
    # plt.plot(x, y_f)
    #
    # plus = mpatches.Patch(color='green', label='f(x) = +1')
    # minus = mpatches.Patch(color='red', label='f(x) = -1')
    # zero = mpatches.Patch(color='blue', label='f(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w_f[0], w_f[1], w_f[2]))
    # plt.legend(handles=[plus, minus, zero])
    # plt.show()

    # w = perceptron(data)
    # y_g = slope(w) * x + intercept(w)
    # df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
    # plt.plot(x, y_f)
    # plt.plot(x, y_g, color = 'cyan')
    # g = mpatches.Patch(color='cyan', label='g(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w[0], w[1], w[2]))
    # plt.legend(handles=[plus, minus, zero, g])
    # plt.show()
    #
    # for count in [20, 100, 1000]:
    #     data = generate_data(count)
    #     fields = {
    #         'x1': [datum[0][1] for datum in data],
    #         'x2': [datum[0][2] for datum in data]
    #     }
    #     df = DataFrame(fields, columns = ['x1', 'x2'])
    #     w = perceptron(data)
    #     y_g = slope(w) * x + intercept(w)
    #     df.plot(x = 'x1', y = 'x2', kind = 'scatter', color = ['red' if datum[1] < 0 else 'green' for datum in data], figsize = (10, 10))
    #     plt.plot(x, y_f)
    #     plt.plot(x, y_g, color = 'cyan')
    #     g = mpatches.Patch(color='cyan', label='g(x) = {:.2f} + {:.2f}x1 + {:.2f}x2 = 0'.format(w[0], w[1], w[2]))
    #     plt.legend(handles=[plus, minus, zero, g])
    #     plt.show()
    rep = 100000

    nu_1 = []
    nu_rand = []
    nu_min = []
    freq_1 = [0] * 11
    freq_rand = [0] * 11
    freq_min = [0] * 11
    for i in range(rep):
        results = experiment()
        nu_1.append(results[0])
        nu_rand.append(results[1])
        nu_min.append(results[2])
        freq_1[results[0]] += 1
        freq_rand[results[1]] += 1
        freq_min[results[2]] += 1

    df = DataFrame({'nu_1': nu_1})
    df.hist(bins = range(12), grid = False)
    plt.show()
    df = DataFrame({'nu_rand': nu_rand})
    df.hist(bins = range(12), grid = False)
    plt.show()
    df = DataFrame({'nu_min': nu_min})
    df.hist(bins = range(12), grid = False)
    plt.show()
    print(freq_1)
    print(freq_rand)
    print(freq_min)

    # for i in range(11):
    #     freq_1[i] /= rep
    #     freq_rand[i] /= rep
    #     freq_min[i] /= rep

    # print(freq_1)
    # print(freq_rand)
    # print(freq_min)

    plot_epsilon(freq_1, rep)
    plot_epsilon(freq_rand, rep)
    plot_epsilon(freq_min, rep)
