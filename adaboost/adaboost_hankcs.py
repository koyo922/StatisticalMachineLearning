import numpy as np
from numpy import log, exp, sign
from utils import load_data, calc_precision_recall, plotROC

from operator import gt, lt

EPS = 1e-5  # 1. 提供min/max的外侧邻点 2. 避免除零错误


class Stump:
    def __init__(self, dim=None, thres=None, op=None):
        # 分类的维度、阈值、方向
        self.dim = dim
        self.thres = thres
        self.op = op
        # 下面的三个参数与Stump本身无关；但是方便Adaboost算法构建
        self.error = float('inf')
        self.y_pred = None
        self.alpha = 0

    def predict(self, x):
        # 输出 -1 / 1
        x = x[:, self.dim] if x.ndim == 2 else x[self.dim]
        return self.op(x, self.thres).astype(int) * 2 - 1

    @staticmethod
    def build(x_train, y_train, weights,
              n_steps=10):
        # 暴力扫描样本的各个维度，各个取值，正反方向；构建效果最好的分类树桩
        bestStump = Stump()
        for dim in range(x_train.shape[1]):  # 遍历x的各个维度

            # # 遍历该维度上的各个阈值 (注意 thres要能取到min/max的外侧)
            # for thres in np.linspace(x_train[:, dim].min() - EPS,
            #                          x_train[:, dim].max() + EPS,
            #                          n_steps + 2):
            # 也可以采用跟hankcs博客一致的方式取thres，方便对照
            step_size = (x_train[:, dim].max() - x_train[:, dim].min()) / n_steps
            for j in range(-1, n_steps + 1):
                thres = x_train[:, dim].min() + j * step_size

                # 遍历尝试两个方向
                for op in (gt, lt):
                    s = Stump(dim, thres, op)
                    s.y_pred = s.predict(x_train)
                    s.error = (s.y_pred != y_train).astype(int) @ weights  # 0/1 loss
                    # 记录下效果最好的分类树桩
                    if s.error < bestStump.error:
                        bestStump = s
        return bestStump


class AdaBoost:
    def __init__(self):
        self.stumps = []

    def fit(self, x_train, y_train, n_estimators=40, verbose=False):
        stumps = []
        m, n = x_train.shape
        # 权重初始化为均匀分布
        weights = np.ones(m) / m
        total_y_pred = np.zeros(m)
        for _ in range(n_estimators):
            # 构建新的树桩
            s = Stump.build(x_train, y_train, weights)
            # 计算它的权重
            s.alpha = 0.5 * log((1 - s.error) / max(s.error, EPS))  # 注意EPS 预防除零错误
            # 放入集合
            stumps.append(s)

            # 更新各样本点的权重
            weights *= exp(-1 * s.alpha * y_train * s.y_pred)
            weights /= weights.sum()
            if verbose:
                print('weights: ', weights)

            # 计算当前总的误差，如果为0就 Early Stopping
            total_y_pred += s.alpha * s.y_pred
            # 注意 sign()
            total_error_rate = (sign(total_y_pred) != y_train).astype(int).mean()
            if verbose:
                print('total_error_rate: ', total_error_rate)
            if total_error_rate < EPS:
                break
        self.stumps = stumps

    def predict_proba(self, x, show_addition=False):
        # 如果是要观察各个stump累加的效果的话，要求x是一个样本
        if show_addition:
            cs = np.cumsum([s.predict(x) * s.alpha for s in self.stumps])
            print('cumsum of stumps: ', cs)
            return cs[-1]
        # 否则要求x是一批样本
        x = x[None, :] if np.ndim(x) == 1 else x
        y_pred = [sum(s.predict(xi) * s.alpha
                      for s in self.stumps)
                  for xi in x]
        return np.array(y_pred)

    def predict(self, x, **kwargs):
        return sign(self.predict_proba(x, **kwargs))


if __name__ == '__main__':

    # # toy data
    # x_train = np.array([[1., 2.1],
    #                     [2., 1.1],
    #                     [1.3, 1.],
    #                     [1., 1.],
    #                     [2., 1.]])
    # y_train = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    # ada = AdaBoost()
    # ada.fit(x_train, y_train, n_estimators=30, verbose=True)
    # print(ada.predict(np.array([0, 0]), show_addition=True))

    # horse colic data
    # http://www.hankcs.com/wp-content/uploads/2016/04/horseColicTraining2.txt
    x_train, y_train = load_data('./horseColicTraining2.txt')
    # http://www.hankcs.com/wp-content/uploads/2016/04/horseColicTest2.txt
    x_test, y_test = load_data('./horseColicTest2.txt')
    ada = AdaBoost()
    for n_estimators in (100, 50, 10):
        ada.fit(x_train, y_train, n_estimators=n_estimators)
        print('==================== n_estimators: ', n_estimators)
        print('Precision & Recall (training): ', calc_precision_recall(ada.predict(x_train), y_train))
        print('Precision & Recall (test): ', calc_precision_recall(ada.predict(x_test), y_test))

    # 讲道理应该画test的曲线，但是 hankcs的博客似乎画的是training；一致方便对照
    print('\n\nshowing ROC curve of Precision & Recall(training) ...')
    plotROC(ada.predict_proba(x_train), y_train)
