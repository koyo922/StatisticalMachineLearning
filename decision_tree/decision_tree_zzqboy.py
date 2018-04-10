import numpy as np
from numpy import log2
from collections import Counter, defaultdict


def shannon_entropy(distribution):
    S = sum(distribution)
    return sum(-v / S * log2(v / S) for v in distribution)


class DecisionTreeNode:
    def __init__(self, feature):
        self.feature = feature
        self.values = {}

    @staticmethod
    def build(x_train, y_train, algo='ID3'):
        assert algo in ('ID3', 'C4.5'), "Only ID3 / C4.5 algorithm supported"
        # 统计各种计数
        y_counter = Counter()  # y的分布
        xi_counter = defaultdict(Counter)  # 样本x的第k维,取值分布
        xiy_counter = defaultdict(Counter)  # 样本x的第k维取值为v时，y的分布
        for x, y in zip(x_train, y_train):
            y_counter[y] += 1
            for k, v in enumerate(x):
                xi_counter[k][v] += 1
                xiy_counter[(k, v)][y] += 1

        # 计算 H(D)
        base_ent = shannon_entropy(y_counter.values())

        # 对x_train中的每一维A计算 info_gain(A) = H(D) - H(D|A)
        info_gain = defaultdict(float)
        for k in range(x_train.shape[1]):
            kv_cnt_sum = sum(xi_counter[k].values())
            # H(D|A)
            cond_ent = sum(kv_cnt / kv_cnt_sum *
                           shannon_entropy(xiy_counter[(k, v)].values())
                           for v, kv_cnt in xi_counter[k].items())
            info_gain[k] = (base_ent - cond_ent) / (1 if algo == 'ID3' else base_ent)

        # 按照info_gain值排序，选择最佳特征，递归构建子树
        info_gain = list(sorted(info_gain.items(), key=lambda x: x[1], reverse=True))
        print('各个特征的 信息增益/信息增益比 为：', info_gain)
        best_feat = info_gain[0][0]
        print('选取最佳特征：', best_feat)

        node = DecisionTreeNode(best_feat)
        # items()的顺序是随机的，可能导致输出顺序不同于
        # https://github.com/zzqboy/static_study/blob/master/decision_tree/decision_tree.py
        # 但是最终生成的树是一致的
        # 注意： 这里要用 best_feat 而非 k
        for v, kv_cnt in xi_counter[best_feat].items():
            if len(xiy_counter[(best_feat, v)]) == 1:
                # 注意 py3中 dict_keys类型不支持方括号取值，改用popitem()
                node.values[v] = xiy_counter[(best_feat, v)].popitem()[0]
                print('特征 {} = 取值 {}, 叶节点 {}'.format(best_feat, v, node.values[v]))
            else:
                mask = x_train[:, best_feat] == v
                node.values[v] = DecisionTreeNode.build(x_train[mask], y_train[mask])
        return node


if __name__ == '__main__':
    x_train = np.array([[1, 2, 2, 1],
                        [1, 2, 2, 2],
                        [1, 1, 2, 2],
                        [1, 1, 1, 1],
                        [1, 2, 2, 1],
                        [2, 2, 2, 1],
                        [2, 2, 2, 2],
                        [2, 1, 1, 2],
                        [2, 2, 1, 3],
                        [2, 2, 1, 3],
                        [3, 2, 1, 3],
                        [3, 2, 1, 2],
                        [3, 1, 2, 2],
                        [3, 1, 2, 3],
                        [3, 2, 2, 1]])
    y_train = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

    dt = DecisionTreeNode.build(x_train, y_train, algo='ID3')
    print()
    dt = DecisionTreeNode.build(x_train, y_train, algo='C4.5')
