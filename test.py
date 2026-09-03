import os
import numpy as np
import torch
from pathlib import Path
from model import SSLBD
from visualization import plot_comparison, load_data
from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from scores import export_metrics_csv


def test_ssl_bd(
    seismic_data: np.ndarray,
    ground_truth: np.ndarray = None,
    base_channels: int = 16,
    device: str = None,
    results_dir: str = None
):
    """
    Executa o teste completo do SSL-BD sobre um dado sísmico 2D.

    Parâmetros
    ----------
    seismic_data : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico observado (Y_obs).
    ground_truth : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico verdadeiro (R_real).
    base_channels : int
        Número de canais da primeira camada da rede de refletividade.
    device : str, opcional
        'cuda' ou 'cpu'. Se None, usa GPU se disponível.
    results_dir: str
        Caminho para salvar resultados
    """

    assert results_dir is not None, "O path para os resultados precisa ser definido"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Passo 1 (Seção 2.2): estimativa do wavelet inicial de fase zero,
    # calculada uma única vez
    # --------------------------------------------------------------

    w0_np = estimate_zero_phase_wavelet(seismic_data)
    w0 = torch.tensor(w0_np, dtype=torch.float32, device=device)

    # dado sísmico como tensor (batch=1, canal=1, n_traces, n_samples)
    y_obs = torch.tensor(seismic_data, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(0).unsqueeze(0)

    if ground_truth is not None:
        r_real = torch.tensor(ground_truth, dtype=torch.float32, device=device)
        r_real = r_real.unsqueeze(0).unsqueeze(0)
    else:
        r_real = None

    # --------------------------------------------------------------
    # Passo 2: Carrega o checkpoint e o modelo.
    # --------------------------------------------------------------

    checkpoint = torch.load(
        f'{results_dir}/best_model.pth',
        map_location=device
    )

    model = SSLBD(w0=w0, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------------
    # Passo 3: Inferência
    # --------------------------------------------------------------

    mu = relative_sparsity_mu(1)
    with torch.no_grad():
        results = model(y_obs, mu=mu)

    return results


is_supervised = True
SUP = 'SUP' if is_supervised else 'S-SUP'
TRAIN_Y_PATH = "/home/data/IN.npy"
TEST_Y_PATH = "/home/data/IN.npy"
X_PATH = None if 'IMG' in TEST_Y_PATH else "/home/data/RFLT.npy"
EPOCHS = 10000
LR = 1e-5
BASE_CHANNELS = 16
RESULTS_DIR = f'/home/src/results/SSLBD_{Path(TRAIN_Y_PATH).stem}_{SUP}_{EPOCHS}_{LR}_{BASE_CHANNELS}'
TEST_DIR = f'{RESULTS_DIR}/test_{Path(TEST_Y_PATH).stem}'
os.makedirs(TEST_DIR, exist_ok=True)

print(f'{SUP}  - Epochs: {EPOCHS} - LR: {LR} - Base Channels: {BASE_CHANNELS}')

y = load_data(data_path=TEST_Y_PATH)
print(f'Imagem blurred: {TEST_Y_PATH}    -  Shape: {y.shape} - min: {y.min()}    - max: {y.max()}')

if X_PATH is not None:
    x = load_data(data_path=X_PATH)
    print(f'Imagem limpa: {X_PATH}    -  Shape: {x.shape} - min: {x.min()}  - max: {x.max()}')
else:
    is_supervised = False
    x = None
results = test_ssl_bd(y, x, base_channels=BASE_CHANNELS, results_dir=RESULTS_DIR)

reflectivity = (results["reflectivity"].detach().cpu().numpy().squeeze())
reflectivity_sparse = (results["reflectivity_sparse"].detach().cpu().numpy().squeeze())
np.save(f'{TEST_DIR}/reflectivity.npy', reflectivity)
np.save(f'{TEST_DIR}/reflectivity_sparse.npy', reflectivity_sparse)

reflectivity_snr2, reflectivity_sparse_snr2 = export_metrics_csv(f'{TEST_DIR}/metrics.csv', y, x, reflectivity, reflectivity_sparse)

#input, output, snr2, results_dir, name, target=None
plot_comparison(input=y, output=reflectivity, snr2=reflectivity_snr2, results_dir=TEST_DIR, name='reflectivity', target=x)
plot_comparison(input=y, output=reflectivity_sparse, snr2=reflectivity_snr2, results_dir=TEST_DIR, name='reflectivity_sparse', target=x)

print('Fim do processamento')

