import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

def plot_images(y, x_hat, cmap, results_dir, name, x=None, epoch=None, snr2=None):
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
            width_ratios=[1, 1, 0.025],
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

    if snr2 is not None:
        ax2.text(0.99, 0.99, 'SNR2: {:.2f}'.format(snr2), horizontalalignment='right', verticalalignment='top', transform=ax2.transAxes, fontsize=12)

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

def plot_comparison(input, output, snr2, results_dir, name, target=None, coord=None):
    """zoom_start_0 = coord[0]
    zoom_start_1 = coord[1]
    zoom_end_0 = coord[2]
    zoom_end_1 = coord[3]
    zoom_start = [zoom_start_0, zoom_start_1]

    rectangle_params = {
        "xy": zoom_start,
        "width": zoom_end_0 - zoom_start_0,
        "height": zoom_end_1 - zoom_start_1,
        "linewidth": 2,
        "edgecolor": 'r',
        "facecolor": 'none'
    }"""
    n_columns = 3
    column_control = 7
    vmin, vmax = -1, 1

    if target is not None:
        n_columns = 4
        column_control = 9
        vmin, vmax = vmin_vmax_percentile(target)


    title_input = 'Input'
    title_target = 'Target'
    title_output = 'Output'
    title_diff = 'Difference'

    def make_subplot(ax, img, title, snr2=None):
        im = ax.imshow(img, cmap="seismic", vmin=vmin, vmax=vmax, aspect='auto')
        """rect = patches.Rectangle(**rectangle_params)

        ax.add_patch(rect)"""
        ax.set_title(title, fontsize=16)
        if title == title_input:
            ax.set_ylabel('Time [ms]')
        if snr2 is not None:
            ax.text(0.99, 0.99, 'SNR2: {:.2f}'.format(snr2), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=16)
        return im

    """def make_zoom(ax, img, title=None):
        slice = (img)[zoom_start_1:zoom_end_1, zoom_start_0:zoom_end_0]
        ax.imshow(slice, cmap="seismic", vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlabel('Trace')
        ax.set_yticks(np.arange(0, zoom_end_1-zoom_start_1, 10))
        ax.set_yticklabels(np.arange(zoom_start_1,zoom_end_1, 10))
        ax.set_xticks(np.arange(0,zoom_end_0-zoom_start_0, 10)) 
        ax.set_xticklabels(np.arange(zoom_start_0,zoom_end_0, 10))
        if title == title_input:
            ax.set_ylabel('Time [ms]')"""

    fig = plt.figure(figsize=(30, 16), facecolor='white')
    fig.subplots_adjust(bottom=0.05, left=0.05, top = 0.975, right=0.975)
    fig.tight_layout()

    item_index = 1

    ax = fig.add_subplot(4, n_columns, (item_index, column_control))
    im = make_subplot(ax, input, title_input)
    column_control+=1
    item_index+=1

    if target is not None:
        ax = fig.add_subplot(4, n_columns, (item_index, column_control))
        im = make_subplot(ax, target, title_target)
        column_control+=1
        item_index+=1

    ax = fig.add_subplot(4, n_columns, (item_index, column_control))
    make_subplot(ax, output, title_output, snr2)
    column_control+=1
    item_index+=1

    if target is not None:
        ax = fig.add_subplot(4, n_columns, (item_index, column_control))
        make_subplot(ax, output - target, title_diff)
        column_control+=1
        item_index+=1   

    """# Zoom
    ax = fig.add_subplot(4, n_columns, column_control)
    make_zoom(ax, input, title_input)
    column_control+=1

    if target is not None:    
        ax = fig.add_subplot(4, n_columns, column_control)
        make_zoom(ax, target)
        column_control+=1

    ax = fig.add_subplot(4, n_columns, column_control)
    make_zoom(ax, output)
    column_control+=1

    if target is not None:
        ax = fig.add_subplot(4, n_columns, column_control)
        make_zoom(ax, output - target)
        column_control+=1"""

    cbar = fig.colorbar(im, ax=fig.axes, pad=0.01, aspect=80)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(f'{results_dir}/{name}.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/{name}.pdf', bbox_inches='tight', transparent=False)

    plt.close()
