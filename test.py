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
    with torch.no_grad():
        outputs = model(y_obs, mu=mu)

    reflectivity = outputs["reflectivity"].detach().cpu().numpy()

    print(reflectivity.shape)
    reflectivity = reflectivity.squeeze()
    print(reflectivity.shape)

    np.save(f'{test_dir}/reflectivity.npy', reflectivity)

    snr2 = export_metrics_csv(input=seismic_data, output=reflectivity, results_dir=test_dir, target=ground_truth)

    plot_comparison(input=seismic_data, output=reflectivity, results_dir=test_dir, name='reflectivity', snr2=snr2, target=ground_truth) 

    return reflectivity


def call_test(is_supervised, train_y_path, epochs, lr, base_channels, test_y_path):
    SUP = 'SUP' if is_supervised else 'SELF-SUP'
    TRAIN_Y_PATH = train_y_path
    TEST_Y_PATH = test_y_path
    X_PATH = f'{Path(test_y_path).parent}/IMG_Ideal.sgy'
    EPOCHS = epochs
    LR = lr
    BASE_CHANNELS = base_channels

    RESULTS_DIR = f'/home/src/results_new_data/SSLBD_DATA_{Path(TRAIN_Y_PATH).stem}_{SUP}_EP_{EPOCHS}_LR_{LR}_BC_{BASE_CHANNELS}'
    TEST_DIR = f'{RESULTS_DIR}/test_{Path(TEST_Y_PATH).stem}'
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f'TESTING: {SUP}  - Epochs: {EPOCHS} - LR: {LR} - Base Channels: {BASE_CHANNELS}')

    y = load_data(data_path=TEST_Y_PATH)
    print(f'Imagem blurred: {TEST_Y_PATH}    -  Shape: {y.shape} - min: {y.min()}    - max: {y.max()}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH)
        print(f'Imagem limpa: {X_PATH}    -  Shape: {x.shape} - min: {x.min()}  - max: {x.max()}')
    else:
        is_supervised = False
        x = None

    test_ssl_bd(y, x, base_channels=BASE_CHANNELS, results_dir=RESULTS_DIR, test_dir=TEST_DIR)    

    print('Fim do teste')