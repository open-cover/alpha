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

    def go(self, row, col):
        row = np.array(row)
        col = np.array(col)
        return row @ self.entries @ col

    def exploitability(self, row_logits, col_logits):
        row_p = softmax(row_logits)
        col_p = softmax(col_logits)

        # current value
        v = row_p @ self.entries @ col_p

        # row best response
        row_payoffs = self.entries @ col_p
        v_row_best = np.max(row_payoffs)

        # col best response
        col_payoffs = row_p @ self.entries
        v_col_best = np.min(col_payoffs)

        return (v_row_best - v) + (v - v_col_best)


class Search:

    def __init__(self, matrix, weight):
        self.matrix = matrix
        self.weight = weight
        self.row = np.zeros(matrix.m)
        self.col = np.zeros(matrix.n)
        self.row_grad = np.zeros(matrix.m)
        self.col_grad = np.zeros(matrix.n)


    def backward(self):

        row_p = softmax(self.row)
        col_p = softmax(self.col)

        v = self.matrix.go(row_p, col_p)

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
            self.row_grad[i] = (row_q[i] - row_v)
        for j in range(self.matrix.n):
            self.col_grad[j] = (col_q[j] - (1 - row_v))


    def update(self, row_lr, col_lr):

        for i in range(self.matrix.m):
            self.row[i] += row_lr * self.row_grad[i]
        for j in range(self.matrix.n):
            self.col[j] = col_lr * self.col_grad[j]

        self.row_grad.fill(0)
        self.col_grad.fill(0)
        

    def alpha(self):
        row_p = softmax(self.row)
        col_payoffs = row_p @ self.matrix.entries
        return np.min(col_payoffs) * self.weight


    def expl(self,):
        return self.matrix.exploitability(self.row, self.col)

def main():
    import sys

    m = 4
    max_n = 9

    np.set_printoptions(precision=3, suppress=True)

    n_searches = 2

    weights = np.random.rand(n_searches)
    weights /= weights.sum()

    searches = []
    # for _ in range(n_searches):
    #     n = np.random.randint(4, max_n)
    #     matrix = Matrix(m, n)
    #     searches.append(Search(matrix, weights[_]))

    # all zeros matrix
    n = 4
    zeros_matrix = np.zeros((m, n))
    row_index = 2
    matrix_with_one_row = np.zeros((m, n))
    matrix_with_one_row[row_index] = 1
    searches.append(Search(Matrix(zeros_matrix), .95))
    searches.append(Search(Matrix(matrix_with_one_row), .05))

    lr = .01
    steps = 1000
    window = 10

    for _ in range(steps):

        for search in searches:
            search.backward()
        for search, w in zip(searches, weights):
            search.update(lr * w, lr)


        if (_ % window) == 0:
            alpha = 0
            for search in searches:
                alpha += search.alpha()
            print(alpha)

if __name__ == "__main__":
    main()
