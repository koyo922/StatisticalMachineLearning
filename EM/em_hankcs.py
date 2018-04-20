import numpy as np
from scipy.stats import binom
from sklearn.exceptions import ConvergenceWarning
from utils import l1_norm  # 参见相应模块的代码


class EmTwoCoins:
    def __init__(self):
        self.θA, self.θB = 0.5, 0.5

    @property
    def θ(self):
        return self.θA, self.θB

    @θ.setter
    def θ(self, *args):
        # 长度为1时，进来的是一个2元数组；长度为2是，进来的是两个float
        self.θA, self.θB = args[0] if len(args) == 1 else args

    def single_iter(self, y, method='vectorized'):
        old_θA, old_θB = self.θ

        if method == 'vectorized':
            N = y.shape[1]
            # 这里通过贝叶斯公式求μA，要对pmf的结果做归一化
            μA = np.apply_along_axis(binom.pmf, 0, y.sum(axis=1), n=N, p=self.θA)
            μB = np.apply_along_axis(binom.pmf, 0, y.sum(axis=1), n=N, p=self.θB)
            # 沿着列做L1-norm，是确保每次实验的 μA + μB = 1；即 μA变量的本意
            μ = l1_norm(np.c_[μA, μB], axis=1)

            # 这里有两种写法，都正确
            # 一是沿着行做 L1-norm，从 μA 求 μA_bar作为行的权重；
            # 用各行"正面率"的加权和作为结果
            # self.θ = l1_norm(μ, axis=0).T @ y.mean(axis=1)

            # 二是不做显式归一化（分子分母上都有，抵消了）；
            # 用带权的各行"正面数" 除以 带权的各行"实验次数"
            self.θ = (μ.T @ y.sum(axis=1)) / (μ.sum(axis=0) * N)
        elif method == 'sample_wise':
            AH, AT, BH, BT = 0, 0, 0, 0
            for y in y:
                N = y.size
                H = y.sum()
                T = N - H
                # E-step
                contribA = binom.pmf(H, N, self.θA)
                contribB = binom.pmf(H, N, self.θB)
                weightA = contribA / (contribA + contribB)
                weightB = contribB / (contribA + contribB)
                AH += weightA * H
                AT += weightA * T
                BH += weightB * H
                BT += weightB * T
            # M-step
            # 这里的weightA并没有沿着j轴做 L1-norm；但是分子分母一除也消去了
            # AH + AT = A(H + T) = AN; 对应于上述第二种写法
            self.θ = AH / (AH + AT), BH / (BH + BT)
        else:
            raise NotImplemented

        return max(abs(self.θA - old_θA), abs(self.θB - old_θB))

    def fit(self, y_train, priors=(0.5, 0.5), ε=1e-6, n_epoch=100, **kwargs):
        self.θ = priors
        for idx_epoch in range(n_epoch):
            if self.single_iter(y_train, **kwargs) < ε:
                return idx_epoch
        raise ConvergenceWarning('Did not converge within {} epochs'.format(n_epoch))
        return n_epoch


if __name__ == '__main__':
    y_train = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    em = EmTwoCoins()

    for priors in [[0.6, 0.5], [0.3, 0.3], [0.9999, 0.00000001]]:
        print('==================== priors: ', priors)
        # idx_epoch = em.fit(y_train, priors, method='sample_wise')
        idx_epoch = em.fit(y_train, priors, method='vectorized', n_epoch=100)
        print('theta: {}, idx_epoch: {}'.format(em.θ, idx_epoch))

"""
==================== priors:  [0.6, 0.5]
theta: (0.7967889544439393, 0.5195834506301285), idx_epoch: 15
==================== priors:  [0.3, 0.3]
theta: (0.66, 0.66), idx_epoch: 1
==================== priors:  [0.9999, 1e-08]
theta: (0.7967888498326234, 0.5195828039168296), idx_epoch: 14
"""
