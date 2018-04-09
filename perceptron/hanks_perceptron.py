import logging

logging.basicConfig(level=logging.INFO)

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

# noinspection PyCallingNonCallable
training_set = np.array([
    (3, 3, 1),
    (4, 3, 1),
    (1, 1, -1)
])

w, b = np.array([0, 0]), 0
history = []


# noinspection PyPep8Naming
def SGD(xi: np.ndarray, yi: np.float) -> None:
    """
    使用误分类点(xi,yi)来执行梯度下降
    :param xi:
    :param yi:
    :return:
    """
    global w, b, history
    w += yi * xi
    b += yi
    logging.info('w=%s, b=%.3f', w, b)
    history.append((w.copy(), b))


def functional_distance(xi, yi) -> np.float:
    """
    计算点(xi,yi)到分类超平面的距离
    即 yi(w*xi+b)
    :param xi:
    :param yi:
    :return:
    """
    global w, b
    return yi * (w @ xi + b)


def train_once():
    """
    扫描一遍数据集，并根据每个误分类点SGD
    :return: 是否发现有误分类点
    """
    error_found = False
    for p in training_set:
        xi, yi = p[:-1], p[-1]
        if functional_distance(xi, yi) <= 0:
            error_found = True
            SGD(xi, yi)

    if not error_found:
        logging.info('RESULT: w: %s, w: %s', w, b)
    return error_found


if __name__ == '__main__':
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

        if w[1] == 0:  # 避免除零
            # if False:
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
    anim.save('perceptron.gif', fps=2, writer='imagemagick')
