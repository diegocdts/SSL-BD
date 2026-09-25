import numpy as np
import segyio
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path


def zScore(data):
    mean = np.mean(data)
    std = np.std(data)
    n_std = 3
    return (data - mean) / (std * n_std)

def load_data(data_path: str, file_name: str = None, to_norm: bool = True):
    path = Path(data_path)
    if path.is_file():
        if '.npy' in data_path:
            data = np.load(data_path).astype("float32")
        else:
            with segyio.open(data_path, ignore_geometry=True) as file:
                data = segyio.collect(file.trace)
        data = data if data.ndim == 2 else data[0]
        data = data.reshape(data.shape[-2], data.shape[-1])
    else:
        assert file_name is not None, "O nome (extensão) do arquivo a ser carregado precisa ser informado"
        data_list = []
        subdirs = [str(p) for p in path.iterdir() if p.is_dir()]
        if '.npy' in file_name:
            for subdir in subdirs:
                file_path = f'{subdir}/{file_name}'
                loaded_data = np.load(file_path).astype("float32")
                data_list.append(loaded_data)
        else:
            for subdir in subdirs:
                file_path = f'{subdir}/{file_name}'
                with segyio.open(file_path, ignore_geometry=True) as file:
                    loaded_data = segyio.collect(file.trace)
                data_list.append(loaded_data)
        data = np.stack(data_list)
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


def plot_wavelet(
    wavelet: np.ndarray,
    results_dir: str,
    dt: float = None,
    true_wavelet: np.ndarray = None,
    title: str = "Wavelet",
    ax: plt.Axes = None,
):
    """
    Plota a wavelet predita (amplitude x tempo), no mesmo estilo dos
    gráficos de wavelet do artigo (ex.: Figuras 6, 12, 17).
 
    Parâmetros
    ----------
    wavelet : np.ndarray ou torch.Tensor, shape (L,)
        Wavelet predita (por exemplo, `resultado['wavelet']` retornado
        por `train_ssl_bd`, ou `outputs['wavelet']` do forward do
        modelo).
    dt : float, opcional
        Intervalo de amostragem no tempo (em segundos). Se fornecido, o
        eixo x é mostrado em segundos, centralizado em torno de zero
        (como nas Figuras 6, 12 e 17). Se None, o eixo x mostra apenas o
        índice da amostra.
    true_wavelet : np.ndarray, opcional
        Wavelet verdadeira (ground truth), se disponível, para sobrepor
        a curva estimada com a curva real -- reproduzindo a comparação
        feita na Figura 6 do artigo ("Estimated" vs "True").
    title : str
        Título do gráfico.
    ax : matplotlib.axes.Axes, opcional
        Eixo onde plotar. Se None, cria uma nova figura.
 
    Retorna
    -------
    matplotlib.axes.Axes
        O eixo com o gráfico plotado.
    """
    # aceita tanto np.ndarray quanto torch.Tensor sem exigir import de torch
    wavelet = np.asarray(
        wavelet.detach().cpu().numpy() if hasattr(wavelet, "detach") else wavelet,
        dtype=float,
    )
 
    n = wavelet.shape[0]
 
    if dt is not None:
        # eixo do tempo centralizado em zero, como nas Figuras 6/12/17
        t = (np.arange(n) - n // 2) * dt
        xlabel = "Time (s)"
    else:
        t = np.arange(n)
        xlabel = "Sample index"
 
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
 
    if true_wavelet is not None:
        true_wavelet = np.asarray(
            true_wavelet.detach().cpu().numpy() if hasattr(true_wavelet, "detach") else true_wavelet,
            dtype=float,
        )
        ax.plot(t, true_wavelet, color="black", linestyle="-", label="True")
        ax.plot(t, wavelet, color="red", linestyle="--", label="Estimated")
        ax.legend()
    else:
        ax.plot(t, wavelet, color="red", linestyle="-", label="Estimated")
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
 
    plt.savefig(f'{results_dir}/wavelet.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/wavelet.pdf', bbox_inches='tight', transparent=False)

    plt.close()


def plot_wavelets(directory="wavelets", cmap="Blues", title: str = "Wavelet"):
    """
    Ordena e plota todas as wavelets armazenadas no diretório.

    Os arquivos devem seguir o padrão:
        wavelet_epoch_X.npy

    Parâmetros
    ----------
    directory : str
        Diretório contendo os arquivos .npy.
    cmap : str
        Colormap do Matplotlib usado para o gradiente de cores.
    """

    # Encontrar arquivos no formato wavelet_epoch_X.npy
    pattern = re.compile(r"wavelet_epoch_(\d+)\.npy$")

    wavelet_files = []

    for filename in os.listdir(directory):
        match = pattern.match(filename)

        if match:
            epoch = int(match.group(1))
            wavelet_files.append((epoch, filename))

    if not wavelet_files:
        raise ValueError(
            f"Nenhum arquivo 'wavelet_epoch_X.npy' encontrado em '{directory}'."
        )

    # Ordenar pela época
    wavelet_files.sort(key=lambda x: x[0])

    # Colormap
    colors = plt.get_cmap(cmap)(
        np.linspace(0.25, 1.0, len(wavelet_files))
    )

    # Criar figura
    plt.figure(figsize=(12, 6))

    for color, (epoch, filename) in zip(colors, wavelet_files):

        filepath = os.path.join(directory, filename)
        wavelet = np.load(filepath)

        plt.plot(
            wavelet,
            color=color,
            linewidth=1.5,
            label=f"Epoch {epoch}"
        )

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(alpha=0.2)

    # Mostrar legenda apenas se não houver muitas épocas
    if len(wavelet_files) <= 20:
        plt.legend()

    plt.tight_layout()
    
    plt.savefig(f'{directory}/wavelet_history.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{directory}/wavelet_history.pdf', bbox_inches='tight', transparent=False)

    plt.close()