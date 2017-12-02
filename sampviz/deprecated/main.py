import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from sampviz import create_joint_axes
from matplotlib.ticker import NullFormatter

nullfmt = NullFormatter()



def animate(i):
    global l, s1, s2, dist1, dist2, xlim, ylim, tax, rax
    # TODO: clean up global variables, yuck

    # set the scatter plot data
    l.set_data(s1[:i], s2[:i])

    s = [s1, s2]
    lim = [xlim, ylim]
    axes = [tax, rax]
    c = ["green", "red"]
    n_bins = 24
    o = ["vertical", "horizontal"]


    for j, ax in enumerate(axes):
        ax.clear()
        ax.xaxis.set_major_formatter(nullfmt)
        ax.yaxis.set_major_formatter(nullfmt)
        if j == 0:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(0, 1)
            x = np.linspace(xlim[0], xlim[1], 100)
            ax.plot(x, dist1.pdf(x), color=c[j])
        else:
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlim(0, 1)
            x = np.linspace(ylim[0], ylim[1], 100)
            ax.plot(dist2.pdf(x), x, color=c[j])

        ax.hist(s[j][:i], bins=np.linspace(lim[j][0], lim[j][1], num=n_bins),
                normed=True, alpha=0.25, color=c[j], orientation=o[j])


# dist
dist1 = scipy.stats.norm(0, 1)
dist2 = scipy.stats.norm(0, 1)

ns = 250

# draw samples
s1 = dist1.rvs(size=ns)
s2 = dist2.rvs(size=ns)

# create axes, fig
jax, tax, rax, fig = create_joint_axes()
xlim = (-5, 5)
ylim = (-5, 5)
c1 = "green"
c2 = "red"

halpha = 0.25

# plot distributions
x = np.linspace(xlim[0], xlim[1], 100)
tax.plot(x, dist1.pdf(x), color=c1)
tax.set_xlim(xlim[0], xlim[1])
tax.set_ylim(0, 1)
y = np.linspace(ylim[0], ylim[1], 100)
rax.plot(dist2.pdf(y), x, color=c2)
rax.set_ylim(ylim[0], ylim[1])
rax.set_xlim(0, 1)

# KDE plot
X, Y = np.meshgrid(x, y)
Z = dist1.pdf(X) * dist2.pdf(Y)

# image
# jax.imshow(Z, cmap="Blues", extent=[xlim[0], xlim[1], ylim[0], ylim[1]], alpha=0.4)
# contourf
cfset = jax.contourf(X, Y, Z, alpha=0.4, cmap="Blues")
cset = jax.contour(X, Y, Z, colors='k', alpha=0.2)
jax.clabel(cset, inline=1, fontsize=10, alpha=0.2)
# sns.kdeplot(dist1.pdf(x), dist2.pdf(y), shade=True, ax=jax)

# init scatter plot
l, = jax.plot([], [], "ko")
jax.set_xlim(xlim[0], xlim[1])
jax.set_ylim(ylim[0], ylim[1])

anim = animation.FuncAnimation(fig, animate, frames=ns, interval=100)

# saving as gif requires imagemagick software
#import os
#os.system("ffmpeg -i C:\\my_path\\animation.mp4 C:\\my_path\\animation.gif")


Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=1800)
# anim.save('figures/normal_normal_mc_sampling.mp4', writer=writer)

plt.show()
