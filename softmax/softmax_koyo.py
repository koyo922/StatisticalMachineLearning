# 注意：如果h5py <= 2.7.1 && numpy >= 1.14.0 会报如下 Warning
# .../anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
# 忽略即可；不必折腾

# 受GFW墙的影响，通过keras正常渠道下载总是不顺利；得借梯子
# brew install proxychains-ng
# vi /usr/local/etc/proxychains.conf # 自己设置代理
# 注意下载到指定位置
# proxychains4 wget https://s3.amazonaws.com/img-datasets/mnist.npz -O ~/.keras/datasets/mnist.npz
from keras.datasets import mnist
from numpy import exp, atleast_2d, argmax, zeros
from utils import l1_norm, append_bias, timed, sample_dataset
from sklearn.preprocessing import label_binarize as one_hot
from sklearn.metrics import confusion_matrix, accuracy_score


class Softmax:
    def __init__(self, k=10):
        self.θ = None
        self.k = k

    # 注意不能写 n_epoch=1e5；因为科学计数法是float型的，无法range()
    def fit(self, x, y, n_epoch=200, λ=0.01, η=1e-4):
        # 用 Batch-GD 训练, 效果应该略差于 SGD
        m = x.shape[0]  # 样本个数
        self.θ = zeros((self.k, x.shape[1]))
        y = one_hot(y, range(self.k))
        for idx_epoch in range(n_epoch):
            if idx_epoch % 10 == 0:
                print('idx_epoch: {}'.format(idx_epoch))
            h = self.predict_proba(x)
            # (h - y.T) 是将公式中的负号移到了括号内
            # (h - y.T) @ x 是对x的各行做线性组合，对照公式理解
            # 注意一定要除以m，否则梯度爆炸；出现 inf
            grad = (h - y.T) @ x / m + λ * self.θ
            self.θ -= η * grad

    def predict_proba(self, x):
        # x中的各样本转置后各占一列，矩阵点乘，exp，l1_norm
        # 结果是每个样本为以列，各行分别表示对应数字的概率
        return l1_norm(exp(self.θ @ atleast_2d(x).T), axis=0)

    def predict(self, x):
        # 输出概率最大的数字
        return argmax(self.predict_proba(x), axis=0)

    def test(self, x, y):
        with timed('predict'):
            y_pred = model.predict(x)
        print('========== accuracy_score = {}'.format(accuracy_score(y, y_pred)))
        print('========== confusion_matrix:')
        print(confusion_matrix(y, y_pred))


if __name__ == '__main__':
    with timed('prepare data'):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # 注意要加上截距（方便模型学到偏置）
        X_train = append_bias(X_train)
        X_test = append_bias(X_test)
        # 与 softmax_wendesi保持类似的数据规模
        X_train, y_train = sample_dataset(X_train, y_train, n=28140, seed=66)
        X_test, y_test = sample_dataset(X_test, y_test, n=13860, seed=66)

    model = Softmax()
    with timed('fit'):
        # 当η=1e-3 也会出现 inf
        model.fit(X_train, y_train, n_epoch=100, η=1e-4)

    model.test(X_test, y_test)

"""
实验结果与shuffle数据时的随机种子有关，0.81~0.90之间波动，相当不稳定
不可与网上的其他代码比较

-------------------- Task: <        prepare data> cost 1.4112s
idx_epoch: 0
idx_epoch: 10
idx_epoch: 20
idx_epoch: 30
idx_epoch: 40
idx_epoch: 50
idx_epoch: 60
idx_epoch: 70
idx_epoch: 80
idx_epoch: 90
-------------------- Task: <                 fit> cost 6.0896s
-------------------- Task: <             predict> cost 0.0165s
========== accuracy_score = 0.8823232323232323
========== confusion_matrix:
[[1302    0    5    1    1    1   11    3   17    0]
 [   0 1477    3    4    0    0    4    2   81    0]
 [  13   12 1216   24   18    0   19   32  130    6]
 [   5    0   21 1230    2   24    7   18   86    8]
 [   1    2    7    3 1265    0   18    1   37   30]
 [  24    1   16   74   21  894   22   20  196   10]
 [  27    5    3    0   25   14 1193    1   11    0]
 [   2    9   20   12   15    0    0 1311   20   18]
 [  19    4    6   27   12    4   15   16 1270    0]
 [  21    3    1   10   94    5    0   92   79 1071]]
 """
