import numpy as np

class Matrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        # initialize with random numbers
        self.entries = np.random.rand(m, n)

    # returns row^T * entries * col
    def go(self, row, col):
        row = np.array(row)
        col = np.array(col)
        return row @ self.entries @ col


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def update(matrix, row, col, lr):


    row_p = softmax(row)
    col_p = softmax(col)

    v = matrix.go(row_p, col_p)

    row_v = 0
    row_q = np.zeros((matrix.m,))

    col_q = np.zeros((matrix.n,))

    for i in range(matrix.m):
        for j in range(matrix.n):
            x = matrix.entries[i, j]
            p = row_p[i]
            q = row_q[j]

            row_v += p * q * x
            row_q[i] += p * x
            col_q[j] += q * (1 - x)

    for i in range(matrix.m):
        row[i] += lr * (row_q[i] - row_v)
    for j in range(matrix.n):
        col[j] += lr * (col_q[j] - (1 - row_v))





def main():
    matrix = Matrix(4, 4)
    print(matrix.entries)

    row = [100, 0, 0, 0]
    col = [0, 0, 0, 100]

    

    print(matrix.go(softmax(row), softmax(col)))


if __name__ == "__main__":
    main()
