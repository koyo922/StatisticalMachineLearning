from collections import Counter


class Bayes:
    def __init__(self):
        self.y_counter = Counter()
        self.xly_counter = Counter()
        self.y_prob = {}
        self.xl_given_y_prob = {}

    def fit(self, X, Y, laplacian=0):
        """
        :param X:
        :param Y:
        :param laplacian: 平滑算子，取0就是极大似然估计；取>0就是贝叶斯估计
        :return:
        """
        self.y_counter = Counter(Y)
        xl_space = [set()] * len(X[0])
        for x, y in zip(X, Y):
            for l, x_l in enumerate(x):
                self.xly_counter[(l, x_l, y)] += 1
                xl_space[l].add(x_l)  # X的第l维可能的取值范围

        self.y_prob = {y: (self.y_counter[y] + laplacian * 1) /
                          (sum(self.y_counter.values()) + laplacian * len(self.y_counter))
                       for y in self.y_counter}

        self.xl_given_y_prob = {(l, x_l, y): (self.xly_counter[(l, x_l, y)] + laplacian * 1) /
                                             (self.y_counter[y] + laplacian * len(xl_space[l]))
                                for l, x_l, y in self.xly_counter}

    def predict(self, x):
        res = {}
        for y, p in self.y_prob.items():
            for l, x_l in enumerate(x):
                p *= self.xl_given_y_prob[(l, x_l, y)]
            res[y] = p

        print('类别\t概率')
        for y, p in sorted(res.items(), key=lambda x: x[1], reverse=True):
            print('{}\t{:.5f}'.format(y, p))


if __name__ == '__main__':
    x_train = [[1, 's'], [1, 'M'], [1, 'M'], [1, 's'], [1, 's'], [2, 's'], [2, 'm'], [2, 'm'],
               [2, 'l'], [2, 'l'], [3, 'l'], [3, 'm'], [3, 'm'], [3, 'l'], [3, 'l']]
    y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

    b = Bayes()
    b.fit(x_train, y_train, laplacian=0)
    b.predict([2, 's'])
