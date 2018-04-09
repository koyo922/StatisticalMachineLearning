import numpy as np
from scipy.spatial.distance import euclidean


class KdTreeNode:
    def __init__(self, point):
        self.point = point
        self.visited = False
        self.left, self.right, self.parent = None, None, None

    def __repr__(self):
        return '({})'.format(self.point)

    @property
    def k(self):
        return self.point.size

    @staticmethod
    def build(data, d):
        """
        对数据集data，按照d为pivot
        切分构建KdTree
        :param data:
        :param d:
        :return:
        """
        data = data[data[:, d].argsort()].copy()
        m = len(data) // 2
        root = KdTreeNode(data[m])
        # print(data, (m, data[m]))

        # del data[m] # np.ndarray 不能直接删元素
        # 总共只有两个维度可用 0/1, 每次取not即可来回翻转(注意取int)
        if m > 0:
            root.left = KdTreeNode.build(data[:m], int(not d))
            root.left.parent = root
        if len(data) >= 3:
            root.right = KdTreeNode.build(data[m + 1:], int(not d))
            root.right.parent = root
        return root

    def find_near_parent(self, target, level):
        d = level % self.k
        if target[d] < self.point[d]:
            return self.left.find_near_parent(target, level + 1) if self.left else self
        else:
            return self.right.find_near_parent(target, level + 1) if self.right else self

    def search(self, target, level):
        global res
        if self.visited:
            return
        else:
            self.visited = True

        # c_dist: 当前node与target的距离
        c_dist = euclidean(self.point, target)
        if c_dist < res['dist']:
            res['point'] = self
            res['dist'] = c_dist

        # 如果当前节点到parent的abs距离(松的阈值) 比 到target还近
        p_node = self.parent  # type: KdTreeNode
        d = level % self.k
        if abs(self.point[d] - p_node.point[d]) < c_dist:
            p_node.search(target, level - 1)
            if p_node.left:
                p_node.left.search(target, level)
            if p_node.right:
                p_node.right.search(target, level)


if __name__ == '__main__':
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    target = np.array([9, 4])

    root = KdTreeNode.build(data, 0)
    s_node = root.find_near_parent(target, 0)
    res = {'point': None, 'dist': float('inf')}
    s_node.search(target, 0)
    print(res)
