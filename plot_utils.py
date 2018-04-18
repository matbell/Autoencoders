def plot_3d_latent_space(epochs, encoded_data, labels):
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(dict(latent1=encoded_data[:, 0], latent2=encoded_data[:, 1], latent3=encoded_data[:, 2],
                           label=labels[0]))
    groups = df.groupby('label')

    fig = plt.figure(figsize=(15, 10))
    ax = Axes3D(fig)

    for name, group in groups:
        ax.scatter(group.latent1, group.latent2, group.latent3, s=30, label=name)

    ax.legend()
    plt.savefig(str(epochs) + ".png")
    plt.close(fig)


def plot_3d_latent_space(encoded_data, labels, ax):
    import pandas as pd

    df = pd.DataFrame(dict(latent1=encoded_data[:, 0], latent2=encoded_data[:, 1], latent3=encoded_data[:, 2],
                           label=labels))

    groups = df.groupby('label')

    for name, group in groups:
        ax.scatter(group.latent1, group.latent2, group.latent3, s=10, label=name)

    ax.legend()
    return ax

def plot_2d_latent_space(encoded_data, labels, ax):
    import pandas as pd

    df = pd.DataFrame(dict(latent1=encoded_data[:, 0], latent2=encoded_data[:, 1], label=labels))
    groups = df.groupby('label')

    for name, group in groups:
        ax.scatter(group.latent1, group.latent2, s=10, label=name)

    ax.legend()
    return ax


def plot_latent_space(data_3d, data_2d, labels):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0], projection="3d")
    plot_3d_latent_space(data_3d, labels, ax0)
    ax1 = plt.subplot(gs[1])
    plot_2d_latent_space(data_2d, labels, ax1)