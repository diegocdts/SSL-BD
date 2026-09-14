import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def zScore(data):
    mean = np.mean(data)
    std = np.std(data)
    n_std = 3
    return (data - mean) / (std * n_std)

def load_data(data_path: str, to_norm: bool = True):
    data = np.load(data_path).astype("float32")
    data = data if data.ndim == 2 else data[0]
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

def plot_comparison(input, output, results_dir, name, snr2=None, target=None):
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

    cbar = fig.colorbar(im, ax=fig.axes, pad=0.01, aspect=80)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(f'{results_dir}/{name}.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/{name}.pdf', bbox_inches='tight', transparent=False)

    plt.close()
