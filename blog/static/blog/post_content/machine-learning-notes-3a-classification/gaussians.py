import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(mpl.patches.Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")


fig = plt.figure(figsize=(8, 8 / 3), constrained_layout=True)
gs = mpl.gridspec.GridSpec(2, 6, figure=fig)

for i in range(3):
    mean_1 = np.array([-0.5, -0.5])
    mean_2 = np.array([0.5, 0.5])
    if i == 0:
        cov_1 = np.array([[0.3, 0.2], [0.2, 0.1]], dtype=np.float32)
        cov_2 = np.array([[0.05, -0.1], [-0.1, 0.6]], dtype=np.float32)
    elif i == 1:
        cov_1 = np.array([[1, 0], [0, 0.1]], dtype=np.float32)
        cov_2 = np.array([[0.2, 0], [0, 0.4]], dtype=np.float32)
    else:
        cov_1 = np.array([[0.2, 0], [0, 0.2]], dtype=np.float32)
        cov_2 = np.array([[0.5, 0], [0, 0.5]], dtype=np.float32)
    # cov_1 /= cov_1.sum()
    # cov_2 /= cov_2.sum()

    ax = fig.add_subplot(1, 3, int(i + 1))

    draw_ellipse(mean_1, cov_1, edgecolor="none", facecolor="magenta", alpha=0.2, ax=ax)
    draw_ellipse(mean_2, cov_2, edgecolor="none", facecolor="turquoise", alpha=0.2, ax=ax)
    draw_ellipse(mean_2, cov_2, edgecolor="none", facecolor="turquoise", alpha=0.1, ax=ax)
    draw_ellipse(mean_1, cov_1, edgecolor="none", facecolor="magenta", alpha=0.1, ax=ax)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        # labelbottom=False,
        # labelleft=False
    )

    xticks = range(-3, 4)
    ax.set_xlim(-3.6, 3.6)
    ax.set_xticks(xticks)
    yticks = range(-3, 4)
    ax.set_ylim(-3.6, 3.6)
    ax.set_yticks(yticks)

    # Grids
    ax.set_axisbelow(True)
    ax.tick_params(which="both", top="off", left="off", right="off", bottom="off", length=0)
    ax.grid(linestyle=":", linewidth=0.5)

    ax.set_aspect(1 / ax.get_data_ratio(), adjustable="box")

    if i == 0:
        ax.set_title("Different covariance matrices", fontsize=12)
    if i == 1:
        ax.set_title("Diagonal covariance matrices", fontsize=12)
    if i == 2:
        ax.set_title("Diagonal with equal variance", fontsize=12)
        # link
        ax.text(
            0.68,
            0.04,
            "cookieblues.github.io",
            fontsize=11,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="dimgrey",
            zorder=5,
        )


plt.tight_layout()
plt.savefig("gaussians.png", bbox_inches="tight")
plt.show()
