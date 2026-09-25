import os
import numpy as np
import torch
from pathlib import Path
from model import SSLBD
from visualization import plot_comparison, load_data, extract_patches, reconstruct_patches
from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from scores import export_metrics_csv


def test_ssl_bd(
    seismic_data: np.ndarray,
    ground_truth: np.ndarray = None,
    base_channels: int = 16,
    device: str = None,
    results_dir: str = None,
    test_dir: str = None
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
        Caminho do modelo salvo
    test_dir: str
            Caminho para salvar teste
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
    y_patches, positions = extract_patches(seismic_data)
    y_obs = torch.tensor(y_patches, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(1).unsqueeze(1)

    # ground truth como tensor (batch=1, canal=1, n_traces, n_samples)
    if ground_truth is not None:
        x_patches, _ = extract_patches(ground_truth)
        r_real = torch.tensor(x_patches, dtype=torch.float32, device=device)
        r_real = r_real.unsqueeze(1).unsqueeze(1)
    else:
        r_real = None

    # --------------------------------------------------------------
    # Passo 2: Carrega o checkpoint e o modelo.
    # --------------------------------------------------------------

    checkpoint = torch.load(
        f'{results_dir}/model.pth',
        map_location=device
    )

    model = SSLBD(w0=w0, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------------------
    # Passo 3: Inferência
    # --------------------------------------------------------------

    mu = relative_sparsity_mu(1)
    output_patchs = []
    with torch.no_grad():
        for y in y_obs:
            outputs = model(y, mu=mu)
            reflectivity = outputs["reflectivity"].detach().cpu().numpy()
            output_patchs.append(reflectivity.squeeze())

    output_patchs = np.array(output_patchs)
    prediction = reconstruct_patches(output_patchs, positions, seismic_data.shape)

    np.save(f'{test_dir}/reflectivity.npy', reflectivity)

    snr2 = export_metrics_csv(input=seismic_data, output=prediction, results_dir=test_dir, target=ground_truth)

    plot_comparison(input=seismic_data, output=prediction, results_dir=test_dir, name='reflectivity', snr2=snr2, target=ground_truth) 

    return prediction


def call_test(base_channels, result_dir, test_y_path, test_x_path = None):
    Y_PATH = test_y_path
    X_PATH = test_x_path

    RESULTS_DIR = result_dir
    TEST_DIR = f'{RESULTS_DIR}/test_{Path(Y_PATH).stem}'
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f'TESTING')

    y = load_data(data_path=Y_PATH)
    print(f'Y: Shape: {y.shape}  - min: {y.min():.4f}    - max: {y.max():.4f}  - Path: {Y_PATH}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH)
        print(f'X: Shape: {x.shape}  - min: {x.min():.4f}    - max: {x.max():.4f}    - Path: {X_PATH}')
    else:
        is_supervised = False
        x = None

    test_ssl_bd(y, x, base_channels=base_channels, results_dir=RESULTS_DIR, test_dir=TEST_DIR)    

    print('Fim do teste')