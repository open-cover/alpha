
import random

class Matrix:

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.entries = []
        for i in range(m):
            self.entries.append([0 for _ in range(self.n)])
            for j in range(n):
                self.entries[i][j] = random.random()

    # returns `row^T * entries * col` e.g. expected reward
    def go(self, row, col):
        r = 0
        for i in range(self.m):
            for j in range(self.n):
                r += row[i] * col[j] * self.entries[i][j]
        return r


def main():

    matrix = Matrix(4, 4)
    for _ in range(matrix.m):
        for j in range(matrix.n):
            print(matrix.entries[_][j])


    row = [1, 0,0,0]
    col = [0,0,0,1]
    print(matrix.go(row, col))
main()