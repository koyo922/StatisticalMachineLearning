import numpy as np
from numpy import ones
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from contextlib import contextmanager
from time import time
from warnings import warn


def sample_dataset(*Ds, n=None, frac=None, replace=False, seed=None):
    # 注意支持多个数据集一起采样；要保持下标一致
    # e.g. X_train, y_train如果一起才的下标不一致就错位了
    Ns = [D.shape[0] for D in Ds]
    assert min(Ns) == max(Ns), '#sample differs in each dataset'
    N = Ns[0]
    if n is None:
        assert frac is not None, 'both n and frac are None!'
        assert 0 < frac <= 1
        n = N * frac
    if n > N and not replace:
        warn('n > N, "replace" forced to True')
        replace = True
    if seed is not None:
        np.random.seed(seed)
    idx = choice(N, n, replace=replace)
    return [D[idx, ...] for D in Ds]


_timed_task_id = 0


@contextmanager
def timed(task_name=None, pattern='-------------------- Task: <{task_name:>20s}> cost {cost:.4f}s'):
    if task_name is None:
        global _timed_task_id  # 注意不是nonlocal
        task_name = 'T_{:03d}'.format(_timed_task_id)
        _timed_task_id += 1
    t0 = time()
    yield
    t1 = time()
    print(pattern.format(task_name=task_name, cost=t1 - t0))


def append_bias(x, flatten_to_2D=True):
    N = x.shape[0]
    if flatten_to_2D:
        x = x.reshape((N, -1))
    return np.c_[ones(N), x]


def l1_norm(x, axis=-1):
    return normalize(x, axis=axis, norm='l1')


def load_data(path: str) -> np.array:
    X, Y = [], []
    with open(path) as f:
        for line in f:
            *x, y = line.split()
            X.append([float(v) for v in x])
            Y.append(float(y))
    return np.array(X), np.array(Y)


def calc_precision_recall(y_pred, y_true):
    TN, FN, FP, TP = 0, 0, 0, 0
    for p, t in zip(y_pred, y_true):
        if p == 1:
            if t == p:
                TP += 1
            else:
                FP += 1
        else:
            if t == p:
                TN += 1
            else:
                FN += 1
    return TP / (TP + FP), TP / (TP + FN)


def plotROC(y_pred, y_true):
    # ROC曲线和confusion_matrix原理并不复杂
    # 但是概念非常纠结，不好理清楚
    cur = (1, 1)  # 初始位于右上角
    y_sum = 0  # AUC累加变量
    # y_true中总共有几个 +1 / -1
    n_positive = (y_true == 1).sum()
    n_negative = y_true.size - n_positive
    # 每一步都要相应地走步，横/纵总路程都是1
    x_step, y_step = 1 / n_negative, 1 / n_positive
    #
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # for _, t in sorted(zip(y_pred, y_true)):
    for t in y_true[y_pred.argsort()]:  # 两者似乎有细微差异，可能python自带的sorted()数值不稳定
        # 从最宽松（右上角）开始的，thres相当于逐渐收小
        if t == 1.0:
            dX, dY = 0, y_step
        else:
            dX, dY = x_step, 0
            y_sum += cur[1]
        ax.plot((cur[0], cur[0] - dX), (cur[1], cur[1] - dY), c='b')
        cur = (cur[0] - dX, cur[1] - dY)
    ax.plot((0, 1), (0, 1), 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis((0, 1, 0, 1))
    plt.show()
    print('AUC: ', y_sum * x_step)
