"""
train.py
=========

Script de treinamento do método SSL-BD, seguindo os passos descritos na
Seção 2.1 (Figura 1) e as configurações experimentais da Seção 3 do
artigo:

    - otimizador Adam, taxa de aprendizado constante de 1e-5;
    - até 10.000 épocas de treinamento;
    - função de perda: erro quadrático entre dado observado e
      reconstruído (Equação 8);
    - parâmetro de esparsidade relativa mu cresce ao longo das épocas
      (Equação 7).

Uso:
    from train import train_ssl_bd
    resultado = train_ssl_bd(seismic_data)  # seismic_data: np.ndarray (Ntr, L)
"""

import numpy as np
import torch
import os
import copy
from pathlib import Path

from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from model import SSLBD
from visualization import plot_comparison, plot_wavelet, plot_wavelets, load_data
from scores import export_metrics_csv


def train_ssl_bd(
    seismic_data: np.ndarray,
    ground_truth: np.ndarray = None,
    is_supervised: bool = False,
    n_epochs: int = 10000,
    learning_rate: float = 1e-5,
    base_channels: int = 16,
    device: str = None,
    verbose_every: int = 500,
    results_dir: str = None,
    wavelet_title: str = None
):
    """
    Executa o treinamento completo do SSL-BD sobre um dado sísmico 2D.

    Parâmetros
    ----------
    seismic_data : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico observado (Y_obs).
    ground_truth : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico verdadeiro (R_real).
    is_supervised:
        True se o modelo deve ser treinado de forma supervisionada. Falso caso contrário.
        Se ground_truth for None, is_supervised = False
    n_epochs : int
        Número máximo de épocas de treinamento (artigo: 10.000).
    learning_rate : float
        Taxa de aprendizado do otimizador Adam (artigo: 1e-5).
    base_channels : int
        Número de canais da primeira camada da rede de refletividade.
    device : str, opcional
        'cuda' ou 'cpu'. Se None, usa GPU se disponível.
    verbose_every : int
        A cada quantas épocas imprimir o valor da perda.
    results_dir: str
        Caminho para salvar resultados
    wavelet_title: str
        Titulo da wavelet

    Retorna
    -------
    dict com:
        'model'        : o modelo SSLBD treinado
        'wavelet'      : wavelet final estimado (np.ndarray)
        'reflectivity' : refletividade final prevista (np.ndarray)
        'loss_history' : lista com o valor da perda a cada época
    """

    assert results_dir is not None, "O path para os resultados precisa ser definido"

    wavelets_dir = f'{results_dir}/wavelets'
    os.makedirs(wavelets_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Passo 1 (Seção 2.2): estimativa do wavelet inicial de fase zero,
    # calculada uma única vez, fora do laço de treinamento.
    # --------------------------------------------------------------
    w0_np = estimate_zero_phase_wavelet(seismic_data)
    w0 = torch.tensor(w0_np, dtype=torch.float32, device=device)

    # dado sísmico obsevado como tensor (batch=1, canal=1, n_traces, n_samples)
    y_obs = torch.tensor(seismic_data, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(0).unsqueeze(0)

    # ground truth como tensor (batch=1, canal=1, n_traces, n_samples)
    if ground_truth is not None:
        r_real = torch.tensor(ground_truth, dtype=torch.float32, device=device)
        r_real = r_real.unsqueeze(0).unsqueeze(0)
    else:
        is_supervised = False
        r_real = None

    # --------------------------------------------------------------
    # Passo 2: treinamento do modelo com otimizador Adam e lr=1e-5, (conforme Seção 3)
    # --------------------------------------------------------------
    model = SSLBD(w0=w0, base_channels=base_channels, ground_truth=r_real, is_supervised=is_supervised).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    lower_loss, best_model_state_dict, best_reflectivity, best_wavelet, window_wavelet = None, None, None, None, None

    for epoch in range(n_epochs):
        r_epoch = epoch / max(n_epochs - 1, 1)
        mu = relative_sparsity_mu(r_epoch)  # Equação 7

        optimizer.zero_grad()
        outputs = model(y_obs, mu=mu)  # passos 2-4 da Seção 2.1
        loss = outputs["loss"]

        # atualização dos parâmetros via retropropagação (passo 4)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if lower_loss is None or loss.item() < lower_loss:
            lower_loss = loss.item()
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_reflectivity = outputs["reflectivity"].detach().cpu().numpy().squeeze()
            best_wavelet = outputs["wavelet"].detach().cpu().numpy()
            window_wavelet = best_wavelet

        if verbose_every and (epoch % verbose_every == 0 or epoch == n_epochs - 1):
            print(f"época {epoch:6d} | perda = {loss.item():.6e} | mu = {mu:.4f}")
            if window_wavelet is not None:
                np.save(f'{wavelets_dir}/wavelet_epoch_{epoch}.npy', window_wavelet)
                window_wavelet = None

    # --------------------------------------------------------------
    # Passo 3: salva melhor modelo e wavelet e refletividade correspondentes
    # --------------------------------------------------------------
    torch.save({"model_state_dict": best_model_state_dict}, f'{results_dir}/model.pth')

    np.save(f'{results_dir}/best_wavelet.npy', best_wavelet)
    np.save(f'{results_dir}/best_reflectivity.npy', best_reflectivity)
    np.save(f'{results_dir}/loss_history.npy', loss_history)

    snr2 = export_metrics_csv(input=seismic_data, output=best_reflectivity, results_dir=results_dir, target=ground_truth)
    plot_comparison(input=seismic_data, output=best_reflectivity, results_dir=results_dir, name='best_reflectivity', snr2=snr2, target=ground_truth)
    plot_wavelets(wavelets_dir, title=wavelet_title)

    return {
        "wavelet": best_wavelet,
        "reflectivity": best_reflectivity,
        "loss_history": loss_history,
    }


def call_train(is_supervised, train_y_path, epochs, lr, base_channels):
    SUP = 'SUP' if is_supervised else 'SELF-SUP'
    Y_PATH = train_y_path
    X_PATH = "/home/data/RFLT.npy" if 'IN.npy' in Y_PATH else None
    EPOCHS = epochs
    LR = lr
    BASE_CHANNELS = base_channels

    RESULTS_DIR = f'/home/src/results/SSLBD_DATA_{Path(Y_PATH).stem}_{SUP}_EP_{EPOCHS}_LR_{LR}_BC_{BASE_CHANNELS}'
    WAVELET_TITLE = f'SSLBD_DATA_{Path(Y_PATH).stem}_{SUP}_{BASE_CHANNELS}'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f'TRAINING: {SUP}  - Epochs: {EPOCHS} - LR: {LR} - Base Channels: {BASE_CHANNELS}')

    y = load_data(data_path=Y_PATH)
    print(f'Imagem blurred: {Y_PATH}    -  Shape: {y.shape} - min: {y.min()}    - max: {y.max()}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH)
        print(f'Imagem limpa: {X_PATH}    -  Shape: {x.shape} - min: {x.min()}  - max: {x.max()}')
    else:
        x = None

    train_ssl_bd(y, x, is_supervised, n_epochs=EPOCHS, learning_rate=LR, base_channels=BASE_CHANNELS, results_dir=RESULTS_DIR, wavelet_title=WAVELET_TITLE)

    print('Fim do treino')
    
    return RESULTS_DIR