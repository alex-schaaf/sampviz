import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

nullfmt = NullFormatter()


def create_joint_axes():
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    # create rectangles
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    fig = plt.figure(1, figsize=(8, 8))  # init figure

    # joint plot
    jax = plt.axes(rect_scatter)

    # top
    tax = plt.axes(rect_histx)

    # right
    rax = plt.axes(rect_histy)

    hax = [tax, rax]
    for ax in hax:
        ax.xaxis.set_major_formatter(nullfmt)
        ax.yaxis.set_major_formatter(nullfmt)

    return jax, tax, rax, fig