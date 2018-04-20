import numpy as np
from numpy.random import multinomial
from sklearn.preprocessing import label_binarize as one_hot


def draw_from(distribution):
    return multinomial(1, distribution).argmax()


def approx(a, b, ε=0.05):
    return np.abs((a - b)).max() < ε


class HMM:
    def __init__(self, pi, A, B):
        self.π = pi
        self.A = A
        self.B = B

    def simulate(self, T, debug=False):
        if debug:
            O = [1, 0, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 2, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 0, 2, 2,
                 0, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 2, 2, 1, 2, 2, 1,
                 2, 2, 2, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 0, 1, 2, 0, 2, 1, 1, 0, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2]
            S = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
            return np.array(O), np.array(S)

        # Observations / States
        O, S = np.zeros(T, dtype=int), np.zeros(T, dtype=int)
        S[0] = draw_from(self.pi)
        for t in range(1, T):
            S[t] = draw_from(self.A[S[t - 1]])
            O[t] = draw_from(self.B[S[t]])
        return O, S

    def forward(self, O):
        N = self.A.shape[0]  # 状态的维度
        T = len(O)

        α = np.empty((N, T))
        # α[s, 0] = pi[s] * B[s, O[0]] for any state s
        α[:, 0] = self.π * self.B[:, O[0]]
        for t in range(1, T):
            # 对任意状态s: α[s,t] = α[:,t-1] @ A[:,s] * B[s,O[t]]
            α[:, t] = α[:, t - 1] @ self.A * self.B[:, O[t]]
        return α

    def backward(self, O):
        N = self.A.shape[0]
        T = len(O)

        β = np.empty((N, T))
        β[:, -1] = 1

        for t in reversed(range(T - 1)):
            # β[s,t] = (A[s,:] * B[:,O[t+1]] * β[:,t+1]).sum()
            #           = A[s,:] @ (B[:,O[t+1]] * β[:,t+1])
            # β[:, t] = (self.A * self.B[:, O[t + 1]] * β[:, t + 1]).sum(axis=1) # 等价
            β[:, t] = self.A @ self.B[:, O[t + 1]] * β[:, t + 1]
        return β

    def fit(self, O, method='baum_welch'):
        done = False
        while not done:
            α = self.forward(O)
            β = self.backward(O)

            # ξ[i,j,t] = α[i,t]*A[i,j]*B[j,O[t+1]]*β[j,t+1] / {numerator}.sum(for i & j)
            N = self.A.shape[0]
            T = len(O)
            ξ = np.empty((N, N, T - 1))  # 注意 ξ不包含最后一帧
            for t in range(T - 1):
                num = α[:, [t]] * self.A * self.B[:, O[t + 1]] * β[:, t + 1]
                denom = num.sum()
                ξ[:, :, t] = num / denom

            # γ[i,t] = α[i,t]*β[i,t] / (α[:,t]@β[:,t])
            # γ 的分母就等于 P(O|λ)，任取一个时刻 α[:,t] @ β[:,t] 即可
            # 注意 (α[:,0]@β[:,0])或者α[:,-1].sum()数值不稳定；最好还是基于逐时刻的α,β来算
            γ = α * β / np.einsum('st,st->t', α, β)
            # 也可以用 γ[i,t] = ξ[i,:,t].sum(axis = 2)来算，不过最后一帧需要特殊处理，麻烦

            new_π = γ[:, 0]
            new_A = ξ.sum(axis=-1) / γ[:, :-1].sum(axis=-1)  # 注意γ的最后一帧不能要
            # B[j,k] = γ[j, O[t]==k].mean(axis=1)
            #        = γ[j,:] @ O.one_hot()[:,k]
            new_B = γ @ one_hot(O, range(self.B.shape[1])) / γ.sum(axis=1).reshape(-1, 1)

            if all(approx(a, b) for a, b in zip([self.π, self.A, self.B], [new_π, new_A, new_B])):
                done = True
            self.π, self.A, self.B = new_π, new_A, new_B

    def decode(self, O, method='viterbi'):
        if method == 'approx':
            α = self.forward(O)
            β = self.backward(O)
            γ = α * β / np.einsum('st,st->t', α, β)
            return None, γ.argmax(axis=0)
        elif method == 'viterbi':
            N = self.A.shape[0]
            T = len(O)

            ψ = np.empty((N, T), dtype=int)  # ψ[:,0] 其实没用到
            δ = np.empty((N, T))
            δ[:, 0] = self.π * self.B[:, O[0]]
            for t in range(1, T):
                # 此处的broadcast原理较复杂，建议画图分析或者用np.einsum()
                transition = δ[:, t - 1].reshape(-1, 1) * self.A * self.B[:, O[t]]
                δ[:, t] = transition.max(axis=0)
                ψ[:, t] = transition.argmax(axis=0)

            # backtrace 回溯找路径
            path = np.empty(T, dtype=int)
            path[T - 1] = δ[:, -1].argmax()
            for t in reversed(range(1, T)):  # 注意没用 ψ[:,0]
                path[t - 1] = ψ[path[t], t]
            return δ, path
        else:
            raise NotImplemented


if __name__ == '__main__':
    π = np.array([0.6, 0.4])  # healthy / fever
    A = np.array([[0.7, 0.3],  # transition: health -> fever
                  [0.4, 0.6]])
    B = np.array([[0.5, 0.4, 0.1],  # emission: normal/cold/dizzy
                  [0.1, 0.3, 0.6]])
    hmm = HMM(π, A, B)

    # 尝试用已知参数的HMM模型对观察序列 [normal, cold, dizzy] 做解码
    O = [0, 1, 2]
    dp, S_pred = hmm.decode(O)
    print('dp matrix:\n', dp)
    print('S_pred: {}, probability: {}'.format(S_pred, dp[S_pred[-1], -1]))

    # 用真实的HMM参数模型来伪造一些数据
    O, S = hmm.simulate(100, debug=True)
    # 然后从一组完全均匀的瞎猜的参数出发，用Baum-Welch算法训练模型
    hmm_guess = HMM(np.array([0.5, 0.5]),
                    np.array([[0.5, 0.5],
                              [0.5, 0.5]]),
                    np.array([[0.3, 0.3, 0.3],
                              [0.3, 0.3, 0.3]]))
    hmm_guess.fit(O)
    # 用训练好的模型，针对训练时用过的观察序列，进行Viterbi解码
    _, S_pred = hmm_guess.decode(O, method='viterbi')
    # 看看预测出来的 *状态* 序列的准确率如何
    print('accuracy: ', (S_pred == S).mean())
