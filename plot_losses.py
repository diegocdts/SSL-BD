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

    plt.figure(figsize=(10, 6))

    for loss_path, model_name in zip(loss_paths, model_names):
        loss = np.load(loss_path)

        plt.plot(loss, label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/model_losses.png', bbox_inches='tight', transparent=False)
    plt.savefig(f'{results_dir}/model_losses.pdf', bbox_inches='tight', transparent=False)


loss_paths = [
    "/home/src/results/SSLBD_DATA_IMG_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy",
    "/home/src/results/SSLBD_DATA_IN_SUP_EP_10000_LR_1e-05_BC_128/loss_history.npy",
    "/home/src/results/SSLBD_DATA_reflectivity_SELF-SUP_EP_10000_LR_1e-05_BC_64/loss_history.npy"
]

model_names = [Path(loss_path).parent.name.replace('EP_10000_LR_1e-05_BC_', '') for loss_path in loss_paths]

results_dir = str(Path(loss_paths[0]).parent.parent)

plot_training_losses(loss_paths, model_names, results_dir)

print("Fim do processamento")