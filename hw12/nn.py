import numpy as np
import random
import math

def init_nn(m, w0):
    first = [[w0(), w0(), w0()] for i in range(m)]
    second = [w0() for i in range(m+1)]
    return (np.array(first), np.array(second)) # m * 3, 1 * (m+1)

def forward(v_x0, w, theta, dt=None):
    (m_w1, v_w2) = w
    v_s1 = np.matmul(m_w1, v_x0)
    # v_x1 = np.insert(np.vectorize(np.tanh)(v_s1), 0, 1.0)
    v_x1 = np.array([1.0] + [np.tanh(s) for s in v_s1], dtype=dt)
    s2 = np.dot(v_w2, v_x1)
    x2 = theta(s2)
    return (v_s1, v_x1, s2, x2)

def get_gradients(data, w, theta, theta_p):
    (m_w1, v_w2) = w
    # e = 0
    m_dedw1 = None
    v_dedw2 = None
    for i in range(len(data)):
        (v_x0, y) = data[i]
        (v_s1, v_x1, s2, x2) = forward(v_x0, w, theta)
        # e += 0.25 * (x2 - y) ** 2
        d2 = 0.5 * (x2 - y) * theta_p(s2)
        # v_dedw2 = np.vectorize(lambda x: d2 * x)(v_x1)
        if i == 0:
            v_dedw2 = np.array([d2 * x for x in v_x1])
        else:
            v_dedw2 += np.array([d2 * x for x in v_x1])
        # v_d1 = np.vectorize(lambda x, y: (1 - math.tanh(x) ** 2) * y * d2)(v_s1, v_w2[1: ])
        v_d1 = np.array([(1 - math.tanh(v_s1[i]) ** 2) * v_w2[i+1] * d2 for i in range(len(v_s1))])
        if i == 0:
            m_dedw1 = np.outer(v_d1, v_x0)
        else:
            m_dedw1 += np.outer(v_d1, v_x0)

    return (m_dedw1, v_dedw2)

def get_gradients_numerical(v_x0, y, w, theta):
    (m_w1, v_w2) = w
    c = 0.0001
    e = (forward(v_x0, w, theta)[3] - y) ** 2
    (m_dedw1, v_dedw2) = init_nn(2, lambda : 0.0)
    for i in range(len(m_w1)):
        for j in range(len(m_w1[i])):
            m_w1[i][j] += c
            m_dedw1[i][j] = ((forward(v_x0, w, theta)[3] - y) ** 2 - e) / 4 / c
            m_w1[i][j] -= c
    for i in range(len(v_w2)):
        v_w2[i] += c
        v_dedw2[i] = ((forward(v_x0, w, theta)[3] - y) ** 2 - e) / 4 / c
        v_w2[i] -= c
    return (m_dedw1, v_dedw2)


def update(w, grad_w, eta):
    (m_w1, v_w2) = w
    (m_dedw1, v_dedw2) = grad_w
    # m_w1 -= eta * m_dedw1
    # v_w2 -= eta * v_dedw2
    for i in range(len(m_w1)):
        for j in range(len(m_w1[i])):
            m_w1[i][j] -= eta * m_dedw1[i][j]
    for i in range(len(v_w2)):
        v_w2[i] -= eta * v_dedw2[i]
    return (eta * m_dedw1, eta * v_dedw2)

def classify(w, x):
    return forward(x, w, lambda x: 1 if x > 0 else -1)[3]

if __name__ == '__main__':
    w = init_nn(2, lambda : 0.25)
    v_x = np.array([1.0, 1.0, 2.0])
    print(get_gradients([(v_x, 1)], w, lambda x: x, lambda x: 1))
    print(get_gradients([(v_x, 1)], w, np.tanh, lambda x: 1 - np.tanh(x) ** 2))
    print(get_gradients_numerical(v_x, 1, w, lambda x: x))
    print(get_gradients_numerical(v_x, 1, w, np.tanh))
