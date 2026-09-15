import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_losses(loss_paths, model_names, results_dir):
    """
    Plota as perdas de treinamento de diferentes modelos.

    Parameters
    ----------
    loss_paths : list[str]
        Lista com os caminhos para os arquivos .npy contendo as losses.
    model_names : list[str]
        Nomes dos modelos, na mesma ordem de loss_paths.
    """

    if len(loss_paths) != len(model_names):
        raise ValueError(
            "loss_paths e model_names devem ter o mesmo tamanho."
        )

    # ============================================================
    # Gráfico completo
    # ============================================================
    plt.figure(figsize=(10, 6))

    for loss_path, model_name in zip(loss_paths, model_names):
        loss = np.load(loss_path)

        plt.plot(loss, label=model_name.replace('reflectivity', 'output'))

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f'{results_dir}/model_losses.png',
        bbox_inches='tight',
        transparent=False
    )
    plt.savefig(
        f'{results_dir}/model_losses.pdf',
        bbox_inches='tight',
        transparent=False
    )

    plt.close()

    # ============================================================
    # Gráfico com zoom: épocas 2000 a 10000
    # ============================================================
    def make_zoom(is_self_sup: bool):
        plt.figure(figsize=(10, 6))

        for loss_path, model_name in zip(loss_paths, model_names):
            if is_self_sup:
                if 'SELF' not in loss_path:
                    continue
            else:
                if 'SELF' in loss_path:
                    continue
            loss = np.load(loss_path)

            start_epoch = 2000
            end_epoch = 10000

            # Inclui a época 10000
            loss_zoom = loss[start_epoch:end_epoch + 1]

            epochs = np.arange(start_epoch, start_epoch + len(loss_zoom))

            plt.plot(epochs, loss_zoom, label=model_name.replace('reflectivity', 'output'))

        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss (Epochs 2000–10000)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        margin = 0.05 * (10000 - 2000)
        plt.xlim(2000 - margin, 10000 + margin)
        plt.ylim(-0.001, 0.1)
        plt.tight_layout()

        zoom_name = 'model_losses_zoom_self-sup' if is_self_sup else 'model_losses_zoom_sup'

        plt.savefig(
            f'{results_dir}/{zoom_name}.png',
            bbox_inches='tight',
            transparent=False
        )
        plt.savefig(
            f'{results_dir}/{zoom_name}.pdf',
            bbox_inches='tight',
            transparent=False
        )

        plt.close()
    make_zoom(is_self_sup=True)
    make_zoom(is_self_sup=False)


loss_paths = [
    "/home/src/results/SSLBD_DATA_IMG_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SUP_EP_10000_LR_1e-05_BC_128/loss_history.npy",
    "/home/src/results/SSLBD_DATA_reflectivity_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy"
]

model_names = [
    Path(loss_path).parent.name.replace(
        'EP_10000_LR_1e-05_BC_', ''
    )
    for loss_path in loss_paths
]

results_dir = str(Path(loss_paths[0]).parent.parent)

plot_training_losses(loss_paths, model_names, results_dir)

print("Fim do processamento")