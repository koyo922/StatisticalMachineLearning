from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# noinspection PyCallingNonCallable
training_set = np.array([
    (3, 3, 1),
    (4, 3, 1),
    (1, 1, -1)
])
N = training_set.shape[0]
x = training_set[:, :-1]
y = training_set[:, -1]

a, b = np.zeros(N), 0
Gram = None
history = []


def calc_gram():
    """
    计算Gram矩阵（各数据点的内积）
    :return:
    """
    g = np.empty((N, N), np.float)
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            g[i][j] = xi @ xj
    return g


# noinspection PyPep8Naming
def SGD(i) -> None:
    """
    使用误分类点(xi,yi)来执行梯度下降
    :param xi:
    :param yi:
    :return:
    """
    global a, b, history
    a[i] += 1
    b += y[i]  # 关键是学a；这里的b只是打印
    history.append((a * y @ x, b))
    logging.info('a=%a, b=%.3f', a, b)


def functional_distance(i) -> np.float:
    """
    计算点(xi,yi)到分类超平面的距离
    即 yi(w*xi+b)
    """
    global a, b
    return y[i] * (a * y @ Gram[i] + b)


def train_once():
    """
    扫描一遍数据集，并根据每个误分类点SGD
    :return: 是否发现有误分类点
    """
    error_found = False
    for i in range(N):
        if functional_distance(i) <= 0:
            error_found = True
            SGD(i)

    if not error_found:
        w = a * y @ x
        logging.info('RESULT: w: %s, w: %s', w, b)
    return error_found


if __name__ == '__main__':
    Gram = calc_gram()
    for epoch in range(1000):
        if not train_once():
            break

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')


    def frame_bg():
        """
        画每一帧的背景
        :return:
        """
        global line
        line.set_data([], [])
        # noinspection PyTypeChecker
        positive = training_set[training_set[:, -1] > 0]
        # noinspection PyTypeChecker
        negative = training_set[training_set[:, -1] <= 0]

        plt.plot(positive[:, :-1], positive[:, -1], 'bo')
        plt.plot(negative[:, :-1], negative[:, -1], 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.xlabel('y')
        plt.title('Perceptron Algorithm by hanks.com')
        return line, label


    def get_y(x):
        """
        给定x，绘制分离线上对应的y
        :param x:
        :return:
        """
        global w, b
        return -(b + w[0] * x) / w[1]


    def animate(frame_idx: int):
        global history, ax, line, label

        global w, b  # get_y 里面用的是global
        w, b = history[frame_idx]

        if w[1] == 0:
            return line, label
        else:
            line.set_data([-7, 7], [get_y(-7), get_y(7)])
            label.set_text(history[frame_idx])
            label.set_position([0, get_y(0)])
            return line, label


    logging.info(history)
    anim = animation.FuncAnimation(
        fig, animate, init_func=frame_bg,
        frames=len(history), interval=1000, repeat=True, blit=True
    )
    plt.show()
    # brew install imagemagick if it compalains "ValueError: outfile must be *.htm or *.html"
    anim.save('perceptron_dual.gif', fps=2, writer='imagemagick')
