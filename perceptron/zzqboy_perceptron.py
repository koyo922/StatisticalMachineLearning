import numpy as np


class Perceptron:
    def __init__(self, x_train, y_train, eta=1, w0=0, b0=0, alpha0=0):
        self.eta = eta
        self.N = x_train.shape[0]
        self.w = np.array([w0] * x_train.shape[1])
        self.b = b0
        self.x_train = x_train
        self.y_train = y_train
        # for dual
        self.alpha = np.array([alpha0] * self.N)
        self._gram = None
        self.history = []

    @staticmethod
    def calc_gram(x):
        N = x.shape[0]
        mtx = np.empty((N, N), np.float)
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                mtx[i, j] = xi @ xj
        return mtx

    @property
    def gram(self):
        if self._gram is None:
            self._gram = self.calc_gram(self.x_train)
        return self._gram

    @property
    def wb(self):
        w = self.alpha * self.y_train @ self.x_train
        b = self.alpha @ self.y_train
        return w, b

    def fit(self, use_dual=False):
        self.history.clear()
        if use_dual:
            gram = self.calc_gram(self.x_train)

        while True:
            error_found = False
            if use_dual:
                for i in range(self.N):
                    while (self.alpha * self.y_train @ gram[i] + self.b) * self.y_train[i] <= 0:
                        error_found = True
                        self.alpha[i] += 1
                        self.b += self.y_train[i]
                        self.history.append((self.x_train[i], self.wb))
            else:
                for x, y in zip(self.x_train, self.y_train):
                    while y * (self.w @ x + self.b) <= 0:
                        error_found = True
                        self.w += y * x
                        self.b += y
                        self.history.append((self.w.copy(), self.b))
            if not error_found:
                break

    def show_history(self, use_dual=False):
        print('误分类点\tw\tb' if use_dual else 'w\tb')
        for h in self.history:
            print(h)


if __name__ == '__main__':
    pctr = Perceptron(np.array([[3, 3], [1, 3], [4, 3]])
                      , np.array([1, -1, 1]))
    print('原始形式')
    pctr.fit()
    pctr.show_history()

    print('\n\n对偶形式')
    pctr.fit(use_dual=True)
    pctr.show_history(use_dual=True)
