import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class Matrix:
    def __init__(self, matrix=None, m=None, n=None):
        if matrix is None:
            self.m = m
            self.n = n
            self.entries = np.random.rand(m, n)
        else:
            self.entries = matrix
            self.m = matrix.shape[0]
            self.n = matrix.shape[1]


class Search:

    def __init__(self, matrix, weight):
        self.matrix = matrix
        self.weight = weight
        self.p2_logits = np.zeros(matrix.n)
        self.p1_gradient = np.zeros(matrix.m)
        self.p2_gradient = np.zeros(matrix.n)

    def backward(self, p1_logits):

        p1_policy = softmax(p1_logits)
        p2_policy = softmax(self.p2_logits)

        value = 0
        p1_q_values = np.zeros(self.matrix.m)
        p2_q_values = np.zeros(self.matrix.n)

        # compute V and both players Q values
        for i in range(self.matrix.m):
            for j in range(self.matrix.n):
                x = self.matrix.entries[i, j]
                p = p1_policy[i]
                q = p2_policy[j]

                value += p * q * x
                p1_q_values[i] += q * x
                p2_q_values[j] += p * (1 - x)

        for i in range(self.matrix.m):
            self.p1_gradient[i] = p1_q_values[i] - value
        for j in range(self.matrix.n):
            self.p2_gradient[j] = p2_q_values[j] - (1 - value)

    def update(self, p1_logits, p1_lr, p2_lr):

        for i in range(self.matrix.m):
            p1_logits[i] += p1_lr * self.p1_gradient[i]
        for j in range(self.matrix.n):
            self.p2_logits[j] += p2_lr * self.p2_gradient[j]

        self.p1_gradient.fill(0)
        self.p2_gradient.fill(0)

    def weighted_alpha(self, p1_logits):
        row_p = softmax(p1_logits)
        col_payoffs = row_p @ self.matrix.entries
        return np.min(col_payoffs) * self.weight


def main():
    import sys

    m = 4
    row = np.zeros(m)

    np.set_printoptions(precision=3, suppress=True)

    searches = []

    # Random searches
    # n_searches = 2
    # max_n = 4
    # weights = np.random.rand(n_searches)
    # weights /= weights.sum()
    # for _ in range(n_searches):
    #     n = np.random.randint(4, max_n + 1)
    #     matrix = Matrix(None, m, n)
    #     print(matrix.entries)
    #     searches.append(Search(matrix, weights[_]))
    # Binary example
    n = 4
    zeros_matrix = np.zeros((m, n))
    row_index = 2
    matrix_with_one_row = np.zeros((m, n))
    matrix_with_one_row[row_index] = 1
    searches.append(Search(Matrix(zeros_matrix), 0.95))
    searches.append(Search(Matrix(matrix_with_one_row), 0.05))

    lr = 0.01
    steps = 10000
    window = 100

    for _ in range(steps):

        for search in searches:
            search.backward(row)
        for search in searches:
            search.update(row, lr * search.weight, lr)

        if (_ % window) == 0:
            alpha = 0
            for search in searches:
                alpha += search.weighted_alpha(row)
            print(f"Alpha: {alpha}")


if __name__ == "__main__":
    main()
