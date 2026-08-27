import numpy as np
import matplotlib.pyplot as plt


def zScore(data):
    mean = np.mean(data)
    std = np.std(data)
    n_std = 3
    return (data - mean) / (std * n_std)

def load_data(data_path: str, to_norm: bool = True):
    data = np.load(data_path).astype("float32")[0]
    data = data.reshape(data.shape[-2], data.shape[-1])
    if to_norm:
        data = zScore(data)

    return data

def vmin_vmax_percentile(target):
    vmin = np.percentile(target, 2)
    vmax = np.percentile(target, 98)

    if np.abs(vmin) > np.abs(vmax):
        vmax = np.abs(vmin)
    else:
        vmin = -np.abs(vmax)
    
    return vmin, vmax

def plot_images(y, x_hat, cmap, results_dir, name, x=None, epoch=None):
    fig = plt.figure(figsize=(15, 5))

    y = y[0, 0] if y.ndim == 4 else y
    x_hat = x_hat[0, 0] if x_hat.ndim == 4 else x_hat

    epoch = f'\nÉpoca: {epoch}' if epoch is not None else ''

    if x is not None:
        # 3 imagens + 1 eixo exclusivo para a colorbar
        gs = fig.add_gridspec(
            1, 4,
            width_ratios=[1, 1, 1, 0.05],
            wspace=0.05
        )

        x = x[0, 0] if x.ndim == 4 else x
        vmin, vmax = vmin_vmax_percentile(x)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        cbar_ax = fig.add_subplot(gs[0, 3])

    else:
        # 2 imagens + 1 eixo exclusivo para a colorbar
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1, 1, 0.05],
            wspace=0.05
        )

        vmin, vmax = -1, 1

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        cbar_ax = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------------
    # Imagem de entrada
    # ------------------------------------------------------------------
    im = ax1.imshow(
        y,
        cmap=cmap,
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    ax1.set_title('Imagem de entrada (y)')
    ax1.axis('off')

    # ------------------------------------------------------------------
    # Imagem recuperada
    # ------------------------------------------------------------------
    ax2.imshow(
        x_hat,
        cmap=cmap,
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    ax2.set_title(f'Imagem recuperada (x_hat){epoch}')
    ax2.axis('off')

    # ------------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------------
    if x is not None:
        ax3.imshow(
            x,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        ax3.set_title('Ground Truth (x)')
        ax3.axis('off')

    # ------------------------------------------------------------------
    # Colorbar em eixo separado
    # ------------------------------------------------------------------
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(
        f'{results_dir}/input_output_{cmap}_{name}.png',
        bbox_inches='tight'
    )

    plt.close()