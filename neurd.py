import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

class Matrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.entries = np.random.rand(m, n)

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
            q = col_p[j]

            row_v += p * q * x
            row_q[i] += q * x
            col_q[j] += p * (1 - x)

    for i in range(matrix.m):
        row[i] += lr * (row_q[i] - row_v)
    for j in range(matrix.n):
        col[j] += lr * (col_q[j] - (1 - row_v))


def main():
    import sys

    np.set_printoptions(precision=3, suppress=True)

    m, n = 4, 4
    matrix = Matrix(m, n)
    print("Payoff matrix:\n", matrix.entries)

    row = np.zeros(m)
    col = np.zeros(n)

    steps = 5000
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    print("Steps: ", steps)
    
    lr = 0.01

    for t in range(steps):
        update(matrix, row, col, lr)

    row_p = softmax(row)
    col_p = softmax(col)

    print("\nFinal row strategy:", row_p)
    print("Final col strategy:", col_p)
    print("Exploitability:", matrix.exploitability(row, col))


if __name__ == "__main__":
    main()
