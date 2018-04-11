import numpy as np
from scipy.stats import binom


class EM_Three_Coin:
    def single_iter(self, y_train, priors):
        thetaA, thetaB = priors
        AH, AT, BH, BT = 0, 0, 0, 0
        for y in y_train:
            N = y.size
            H = y.sum()
            T = N - H
            # E-step
            contribA = binom.pmf(H, N, thetaA)
            contribB = binom.pmf(H, N, thetaB)
            weightA = contribA / (contribA + contribB)
            weightB = contribB / (contribA + contribB)
            AH += weightA * H
            AT += weightA * T
            BH += weightB * H
            BT += weightB * T
        # M-step
        return AH / (AH + AT), BH / (BH + BT)

    def fit(self, y_train, priors, eps=1e-6, n_epoch=10000):
        idx_epoch = 0
        while idx_epoch < n_epoch:
            new_priors = self.single_iter(y_train, priors)
            if abs(new_priors[0] - priors[0]) < eps:
                break
            priors = new_priors
            idx_epoch += 1
        return new_priors, idx_epoch


if __name__ == '__main__':
    y_train = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

    em = EM_Three_Coin()

    for priors in [[0.6, 0.5], [0.3, 0.3], [0.9999, 0.00000001]]:
        print('==================== priors: ', priors)
        theta, idx_epoch = em.fit(y_train, priors)
        print('theta: {}, idx_epoch: {}'.format(theta, idx_epoch))
