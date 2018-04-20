from numpy import sign, atleast_2d, matrix, zeros
from utils import append_bias


class Perceptron:
    def __init__(self, dual=True):
        self.dual = dual
        self.θ = None  # 相当于増广了b之后的w
        self.α = None
        self.history = []

    def fit(self, x, y, η=1.0, n_epoch=1000):
        self.history.clear()

        x = append_bias(atleast_2d(x))
        m, n = x.shape
        self.θ = zeros(n)
        if self.dual:
            G = x @ x.T  # Gram 矩阵G
            self.α = zeros(m)
        for idx_epoch in range(n_epoch):
            if self.dual:
                # G 中每一列对应一个样本，点乘完之后得到行向量(其实是1D)
                # 下面这行直接调用self.predict(x)也正确（每轮要多算一下θ）
                # 但是为了强调对偶形式值只需用到Gram矩阵，还是手动计算了
                h = sign(self.α * y @ G)
                grad = -(h != y).astype(int)
                self.α -= η * grad
                # 对偶形式下 θ 不是必需，下面这行与对偶形式的核心逻辑无关
                # 只是为了方便统一记录θ的history，对照观察原始形式的过程
                self.θ = self.α * y @ x
            else:
                h = self.predict(x)
                grad = -(h != y).astype(int) * y @ x
                self.θ -= η * grad

            self.history.append(self.θ.copy())
            if (grad == 0).all():  # 每个样本/特征维度上的梯度均为零
                print('Early Stopped at idx_epoch: {}'.format(idx_epoch))
                break

    def predict(self, x):
        # 即使是dual形式的，也不方便把整个Gram矩阵存下来；最后还是写成θ的形式
        return sign(atleast_2d(x) @ self.θ)

    def show_history(self):
        print('history of θ:')
        for h in self.history:
            print(h)


if __name__ == '__main__':
    x_train, y_train = matrix('3 3;1 3;4 3').A, matrix('1 -1 1').A1

    print('原始形式')
    pctr = Perceptron(dual=False)
    pctr.fit(x_train, y_train)
    pctr.show_history()

    print('\n\n对偶形式')
    pctr = Perceptron(dual=True)
    pctr.fit(x_train, y_train)
    pctr.show_history()

"""
注意偏置在第一列
结果相当于 w=(4, -3), b = -1
与zzqboy的版本一致

原始形式
Early Stopped at idx_epoch: 3
history of θ:
[1. 6. 3.]
[0. 5. 0.]
[-1.  4. -3.]
[-1.  4. -3.]


对偶形式
Early Stopped at idx_epoch: 3
history of θ:
[1. 6. 3.]
[0. 5. 0.]
[-1.  4. -3.]
[-1.  4. -3.]
"""
