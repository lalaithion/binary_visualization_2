import numpy as np
import matplotlib.pyplot as plt


def plot_1d(array, saveto=""):
    """
    Given a 1 dimensional numpy array, this function plots and displays
    a bar graph of the most common elements of the array. The indended use is
    to display the objects (or slices of the objects) returned from the load
    module.

    array: A 1 dimensional numpy array.
    """

    xs = np.arange(0, 256)

    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)
    ax.bar(
        xs,
        array,
        align="center",
        width=1,
        color=plt.cm.jet(array)
    )
    ax.set_yscale("log")

    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()


def plot_2d(array, saveto=""):
    """
    Given a 2 dimensional numpy array, this function plots and displays
    a heatmap of the most common elements of the array. The indended use is
    to display the objects (or slices of the objects) returned from the load
    module.

    array: A 2 dimensional numpy array.
    """
    fig, ax = plt.subplots()

    ax.imshow(
        array,                    # plot the array of values
        interpolation='nearest',  # use neartest neighbor interpolation
        origin='lower',           # put the origin in the lower left
        cmap='jet')

    plt.tick_params(
        axis='both',              # both axis are affected
        which='both',             # both major and minor ticks are affected
        bottom='off',             # ticks along the bottom edge are off
        top='off',                # ticks along the top edge are off
        left='off',               # ticks along the left edge are off
        right='off')              # ticks along the right edge are off

    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()


def plot_3d(array, saveto=""):
    """
    Given a 3 dimensional numpy array, this function plots and displays
    a heatmap of the most common elements of the array. The indended use is
    to display the objects (or slices of the objects) returned from the load
    module.

    array: A 3 dimensional numpy array.
    """
    raise Exception("Unimplemented: plot_3d has not been implemented yet!")


def plot(array, dims=2, saveto=""):
    if dims == 1:
        plot_1d(array, saveto=saveto)
    elif dims == 2:
        plot_2d(array, saveto=saveto)
    elif dims == 3:
        plot_3d(array, saveto=saveto)
