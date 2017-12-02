import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.animation as animation
import numpy as np


class SampViz:
    def __init__(self, dist1, dist2, figsize=(6, 6)):
        self.dist1 = dist1  # scipy.stats distribution object
        self.dist2 = dist2

        self.c1 = "green"
        self.c2 = "red"
        self.f_std = 4
        self.n_bins = 24

        self.d1_e = self.dist1.stats

        self.jax, self.tax, self.rax, self.fig = create_joint_axes(figsize)
        self.axes = [self.jax, self.tax, self.rax]
        # self.l = self.init_scatter(self.jax)

    def sample_monte_carlo(self, n):
        """Create n independent samples from the two distributions."""
        s1 = self.dist1.rvs(size=n)
        s2 = self.dist2.rvs(size=n)
        return np.stack((s1, s2), axis=1)

    def plot_pdf1(self):
        x = np.linspace(-self.dist1.std() * self.f_std, self.dist1.std() * self.f_std, 100)
        self.tax.plot(x, self.dist1.pdf(x), color=self.c1)
        self.tax.set_xlim(-self.dist1.std() * self.f_std, self.dist1.std() * self.f_std)
        self.tax.set_ylim(0, 1)

    def plot_pdf2(self):
        y = np.linspace(-self.dist2.std() * self.f_std, self.dist2.std() * self.f_std, 100)
        self.rax.plot(self.dist2.pdf(y), y, color=self.c2)
        self.rax.set_ylim(-self.dist2.std() * self.f_std, self.dist2.std() * self.f_std)
        self.rax.set_xlim(0, 1)

    def plot_hist1(self, samples):
        self.tax.hist(samples[:, 0], bins=np.linspace(-self.dist1.std() * self.f_std, self.dist1.std() * self.f_std, num=self.n_bins),
                      normed=True, alpha=0.25, color=self.c1)

    def plot_hist2(self, samples):
        self.rax.hist(samples[:, 1],
                      bins=np.linspace(-self.dist2.std() * self.f_std, self.dist2.std() * self.f_std, num=self.n_bins),
                      normed=True, alpha=0.25, color=self.c2, orientation="horizontal")

    def plot_scatter(self, samples, marker=".", color="black", alpha=1):
        self.jax.scatter(samples[:, 0], samples[:, 1], marker=marker, color=color, alpha=alpha)
        self.jax.set_xlim(-self.dist1.std() * self.f_std, self.dist1.std() * self.f_std)
        self.jax.set_ylim(-self.dist2.std() * self.f_std, self.dist2.std() * self.f_std)

    def plot_kde(self, lines=True, cmap="Blues", alpha=0.4):
        x = np.linspace(-self.dist1.std() * self.f_std, self.dist1.std() * self.f_std, 100)
        y = np.linspace(-self.dist2.std() * self.f_std, self.dist2.std() * self.f_std, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.dist1.pdf(X) * self.dist2.pdf(Y)

        cfset = self.jax.contourf(X, Y, Z, alpha=alpha, cmap=cmap)
        if lines:
            cset = self.jax.contour(X, Y, Z, colors='k', alpha=alpha/2.)
            self.jax.clabel(cset, inline=1, fontsize=10, alpha=alpha/2.)

    def jointplot(self, samples, show=True):
        self.plot_kde(lines=False)
        self.plot_scatter(samples)

        self.plot_pdf1()
        self.plot_hist1(samples)

        self.plot_pdf2()
        self.plot_hist2(samples)

        if show:
            plt.show()

    def _animate(self, i, samples, nothing):
        for j, ax in enumerate(self.axes):
            ax.clear()
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

            self.jointplot(samples[:i,:], show=False)
            self.rax.set_title(str(i))

    def run(self, samples, interval=100, save=None):
        ani = animation.FuncAnimation(self.fig, self._animate, frames=samples.shape[0],
                                    fargs=(samples, None), interval=interval)
        plt.show()

        if type(save) is str:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=1800)

            ani.save(save, writer=writer)


def create_joint_axes(figsize):
    """Generates the joint-plot axes and figure.

    Args:
        figsize (:obj:`tuple`): Figure size

    Returns:
        :obj:`tuple`: containing the three different axes and figure.
    """
    nullfmt = NullFormatter()
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    # create rectangles
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    fig = plt.figure(1, figsize=figsize)  # init figure

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