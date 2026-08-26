import matplotlib.pyplot as plt

def plot_images(y, x_hat, cmap, results_dir, name, x=None, epoch=None):
    fig = plt.figure(figsize=(15, 5))

    y = y[0,0] if y.ndim == 4 else y
    x_hat = x_hat[0,0] if x_hat.ndim == 4 else x_hat

    epoch = f'\nÉpoca: {epoch}' if epoch is not None else ''

    if x is not None:
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        x = x[0,0] if x.ndim == 4 else x
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.imshow(y, cmap=cmap, aspect='auto')
    ax1.set_title('Imagem de entrada (y)')
    ax1.axis('off')

    ax2.imshow(x_hat, cmap=cmap, aspect='auto')
    ax2.set_title(f'Imagem recuperada (x_hat){epoch}')
    ax2.axis('off')

    if x is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(x, cmap=cmap, aspect='auto')
        ax3.set_title('Ground Truth (x)')
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(
        f'{results_dir}/input_output_{cmap}_{name}.png',
        bbox_inches='tight'
    )
    plt.close()