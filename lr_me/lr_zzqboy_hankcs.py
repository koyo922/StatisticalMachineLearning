import numpy as np
import sklearn.utils
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LR:
    def __init__(self, init_eta=2, eta=0.001):
        """
        :param init_eta: 逐行训练时采用动态学习率，这里设置初始值
        :param eta: 整体训练时，设置一个固定学习率即可
        """
        self.w = None
        self.init_eta = init_eta
        self.eta = eta

    def fit(self, x_train, y_train, n_epoch=500, style='sample_wise', warm_start=False):
        m, n = x_train.shape

        if style == 'sample_wise':
            w = self.w if warm_start else np.ones(n)
            for idx_epoch in range(n_epoch):
                # 注意这里同时shuffle x和y的技巧
                for idx_sample, (x, y) in enumerate(zip(*sklearn.utils.shuffle(x_train, y_train, random_state=0))):
                    # 动态学习率
                    eta = self.init_eta / (1 + idx_epoch + idx_sample) + 0.01
                    h = sigmoid(x @ w)
                    w += eta * (y - h) * x
        elif style == 'batch':
            w = self.w if warm_start else np.ones((n, 1))
            for idx_epoch in range(n_epoch):
                h = sigmoid(x_train @ w)
                w += self.eta * x_train.T @ (y_train.reshape(-1, 1) - h)
        else:
            raise AttributeError('style should not be: {}'.format(style))

        self.w = w

    def predict_proba(self, x):
        return sigmoid(x @ self.w)

    def predict(self, x, thres=0.5):
        return (self.predict_proba(x) >= thres).astype(int)

    def test(self, x_test, y_test):
        y_pred = self.predict(x_test)
        print('w=', self.w)
        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def plotBestFit(self, x, y):
        """
        画出数据集和逻辑斯谛最佳回归直线
        """
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(x.shape[0]):
            if int(y[i]) == 1:
                xcord1.append(x[i, 1])
                ycord1.append(x[i, 2])
            else:
                xcord2.append(x[i, 1])
                ycord2.append(x[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        if self.w is not None:
            weights = self.w.flatten()
            x = np.arange(-3.0, 3.0, 0.1)
            y = (-weights[0] - weights[1] * x) / weights[2]  # 令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
            ax.plot(x, y)  # 一个作为X一个作为Y，画出直线
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


# 数据下载自 http://www.hankcs.com/wp-content/uploads/2015/09/testSet.txt
def load_data(path='./testSet.txt'):
    X, Y = [], []
    with open(path, 'r') as f:
        for line in f:
            *x, y = line.split()
            # 注意x増广到齐次(否则没有bias)
            X.append([1] + [float(v) for v in x])
            Y.append(float(y))
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    x, y = load_data()

    lr = LR()

    print('====================sample_wise training')
    lr.plotBestFit(x, y)  # 画出数据点（此时还没有weights）
    lr.fit(x, y, style='sample_wise')
    lr.test(x, y)
    lr.plotBestFit(x, y)  # 画出训练好的 weights 分离线

    print('====================batch training')
    lr.fit(x, y, style='batch')
    lr.test(x, y)
    lr.plotBestFit(x, y)

    print('done')

"""
====================sample_wise training
w= [12.14791333  0.99240375 -1.61735944]
0.95
             precision    recall  f1-score   support

        0.0       0.96      0.94      0.95        47
        1.0       0.94      0.96      0.95        53

avg / total       0.95      0.95      0.95       100

====================batch training
w= [[ 4.12414349]
 [ 0.48007329]
 [-0.6168482 ]]
0.96
             precision    recall  f1-score   support

        0.0       0.92      1.00      0.96        47
        1.0       1.00      0.92      0.96        53

avg / total       0.96      0.96      0.96       100

done
"""
