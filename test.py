import os
import numpy as np
import torch
from pathlib import Path
from model import SSLBD
from visualization import plot_comparison2, load_data, extract_patches, reconstruct_patches, crop_data
from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from scores import export_metrics_csv


def embedding(x):
    x = x.float()
    B, C, H, W = x.shape
    device = x.device

    xs = torch.linspace(0, 1, W, device=device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    ys = torch.linspace(0, 1, H, device=device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)

    x = torch.cat([x, xs, ys], dim=1)  # (B,3,H,W)
    return x


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

    # y_start, x_start = 200, 3500
    # dado sísmico como tensor (batch=1, canal=1, n_traces, n_samples)
    # y_patches, positions = extract_patches(seismic_data)
    # y_patch, positions = extract_patches(seismic_data, y_start, x_start, to_norm=True)
    y_obs = torch.tensor(seismic_data, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(0).unsqueeze(0)
    y_obs = embedding(y_obs)

    # ground truth como tensor (batch=1, canal=1, n_traces, n_samples)
    if ground_truth is not None:
        # x_patches, _ = extract_patches(ground_truth)
        # x_patch, _ = extract_patches(ground_truth, y_start, x_start, to_norm=True)
        r_real = torch.tensor(ground_truth, dtype=torch.float32, device=device)
        r_real = r_real.unsqueeze(0).unsqueeze(0)
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
        outputs = model(y_obs, mu=mu)
        prediction = outputs["reflectivity"].detach().cpu().numpy().squeeze()

    np.save(f'{test_dir}/reflectivity.npy', prediction)

    snr2 = export_metrics_csv(input=seismic_data, output=prediction, results_dir=test_dir, target=ground_truth)

    plot_comparison2(input=seismic_data, 
                        output=prediction, 
                        results_dir=test_dir, 
                        name='reflectivity', 
                        snr2=snr2, 
                        target=ground_truth) 
    
    return prediction


def call_test(base_channels, result_dir, test_y_path, test_x_path = None):
    Y_PATH = test_y_path
    X_PATH = test_x_path

    RESULTS_DIR = result_dir
    TEST_DIR = f'{RESULTS_DIR}/test_{Path(Y_PATH).stem}'
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f'TESTING')

    y = load_data(data_path=Y_PATH, to_norm=True)
    print(f'Y: Shape: {y.shape}  - min: {y.min():.4f}    - max: {y.max():.4f}  - Path: {Y_PATH}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH, to_norm=True)
        print(f'X: Shape: {x.shape}  - min: {x.min():.4f}    - max: {x.max():.4f}   - Path: {X_PATH}')
    else:
        x = None

    test_ssl_bd(y, x, base_channels=base_channels, results_dir=RESULTS_DIR, test_dir=TEST_DIR)    

    print('Fim do teste')