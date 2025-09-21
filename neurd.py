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

    def compute_value(self, row, col):
        row = np.array(row)
        col = np.array(col)
        return row @ self.entries @ col


class Search:

    def __init__(self, matrix, weight):
        self.matrix = matrix
        self.weight = weight
        self.col = np.zeros(matrix.n)
        self.row_grad = np.zeros(matrix.m)
        self.col_grad = np.zeros(matrix.n)

    def backward(self, row):

        row_p = softmax(row)
        col_p = softmax(self.col)

        v = self.matrix.compute_value(row_p, col_p)

        row_v = 0
        row_q = np.zeros(self.matrix.m)
        col_q = np.zeros(self.matrix.n)

        for i in range(self.matrix.m):
            for j in range(self.matrix.n):
                x = self.matrix.entries[i, j]
                p = row_p[i]
                q = col_p[j]

                row_v += p * q * x
                row_q[i] += q * x
                col_q[j] += p * (1 - x)

        for i in range(self.matrix.m):
            self.row_grad[i] = row_q[i] - row_v
        for j in range(self.matrix.n):
            self.col_grad[j] = col_q[j] - (1 - row_v)

    def update(self, row, row_lr, col_lr):

        for i in range(self.matrix.m):
            row[i] += row_lr * self.row_grad[i]
        for j in range(self.matrix.n):
            self.col[j] += col_lr * self.col_grad[j]

        self.row_grad.fill(0)
        self.col_grad.fill(0)

    def weighted_alpha(self, row):
        row_p = softmax(row)
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
    searches.append(Search(Matrix(zeros_matrix), .95))
    searches.append(Search(Matrix(matrix_with_one_row), .05))

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
