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
from pathlib import Path

from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from model import SSLBD
from vizualization import plot_images, load_data


def train_ssl_bd(
    seismic_data: np.ndarray,
    ground_truth: np.ndarray = None,
    is_supervised: bool = False,
    n_epochs: int = 10000,
    learning_rate: float = 1e-5,
    base_channels: int = 16,
    device: str = None,
    verbose_every: int = 500,
    results_dir: str = None
):
    """
    Executa o treinamento completo do SSL-BD sobre um dado sísmico 2D.

    Parâmetros
    ----------
    seismic_data : np.ndarray, shape (n_traces, n_samples)
        Dado sísmico observado (Y_obs).
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

    Retorna
    -------
    dict com:
        'model'        : o modelo SSLBD treinado
        'wavelet'      : wavelet final estimado (np.ndarray)
        'reflectivity' : refletividade final prevista (np.ndarray),
                          já com o processo de esparsidade relativa
        'loss_history' : lista com o valor da perda a cada época
    """

    assert results_dir is not None, "O path para os resultados precisa ser definido"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Passo 1 (Seção 2.2): estimativa do wavelet inicial de fase zero,
    # calculada uma única vez, fora do laço de treinamento.
    # --------------------------------------------------------------
    w0_np = estimate_zero_phase_wavelet(seismic_data)
    w0 = torch.tensor(w0_np, dtype=torch.float32, device=device)

    # dado sísmico como tensor (batch=1, canal=1, n_traces, n_samples)
    y_obs = torch.tensor(seismic_data, dtype=torch.float32, device=device)
    y_obs = y_obs.unsqueeze(0).unsqueeze(0)

    r_real = torch.tensor(ground_truth, dtype=torch.float32, device=device)
    r_real = r_real.unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------------------
    # Modelo e otimizador (Adam, lr=1e-5, conforme Seção 3)
    # --------------------------------------------------------------
    model = SSLBD(w0=w0, base_channels=base_channels, ground_truth=r_real, is_supervised=is_supervised).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    lower_loss = None
    best_model = None
    best_reflectivity = None
    best_wavelet = None

    for epoch in range(n_epochs):
        r_epoch = epoch / max(n_epochs - 1, 1)
        mu = relative_sparsity_mu(r_epoch)  # Equação 7

        optimizer.zero_grad()
        outputs = model(y_obs, r_epoch=r_epoch, mu=mu)  # passos 2-4 da Seção 2.1
        loss = outputs["loss"]

        # atualização dos parâmetros via retropropagação (passo 4)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose_every and (epoch % verbose_every == 0 or epoch == n_epochs - 1):
            print(f"época {epoch:6d} | perda = {loss.item():.6e} | mu = {mu:.4f}")

        if lower_loss is None or loss < lower_loss:
            lower_loss = loss
            best_model = model.state_dict()
            y = y_obs.detach().cpu().numpy()
            best_reflectivity = outputs["reflectivity"].detach().cpu().numpy()
            best_wavelet = outputs["wavelet"].detach().cpu().numpy()
            plot_images(y=y, x_hat=best_reflectivity, x=ground_truth, cmap='seismic', results_dir=RESULTS_DIR, name='lower-loss', epoch=epoch)
        

    # --------------------------------------------------------------
    # Passo 5: saída final (wavelet e refletividade) após convergência
    # --------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        r_epoch_final = 1.0
        mu_final = relative_sparsity_mu(r_epoch_final)
        final_outputs = model(y_obs, r_epoch=r_epoch_final, mu=mu_final)

    final_wavelet = final_outputs["wavelet"].detach().cpu().numpy()
    final_reflectivity = (
        final_outputs["reflectivity_sparse"].detach().cpu().numpy().squeeze()
    )

    torch.save({"model_state_dict": best_model}, f'{results_dir}/best_model.pth')
    torch.save({"model_state_dict": model.state_dict}, f'{results_dir}/final_model.pth')

    np.save(f'{results_dir}/best_reflectivity.npy', best_reflectivity)
    np.save(f'{results_dir}/best_wavelet.npy', best_wavelet)
    np.save(f'{results_dir}/final_reflectivity.npy', final_reflectivity)
    np.save(f'{results_dir}/final_wavelet.npy', final_wavelet)
    np.save(f'{results_dir}/loss_history.npy', loss_history)

    y = y_obs.detach().cpu().numpy()
    x = r_real.detach().cpu().numpy()
    plot_images(y=y, x_hat=final_reflectivity, x=x, cmap='seismic', results_dir=results_dir, name='final')

    return {
        "model": model,
        "wavelet": final_wavelet,
        "reflectivity": final_reflectivity,
        "loss_history": loss_history,
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    # ------------------------------------------------------------------
    # Configurações
    # ------------------------------------------------------------------
    is_supervised = False
    SUP = 'SUP' if is_supervised else 'S-SUP'
    Y_PATH = "/home/data/IN.npy"
    X_PATH = "/home/data/RFLT.npy"
    EPOCHS = 10000
    LR = 1e-5
    BASE_CHANNELS = 32
    RESULTS_DIR = f'/home/src/results/SSLBD_{Path(Y_PATH).stem}_{SUP}_{EPOCHS}_{LR}_{BASE_CHANNELS}'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f'{SUP}  - Epochs: {EPOCHS} - LR: {LR} - Base Channels: {BASE_CHANNELS}')

    y = load_data(data_path=Y_PATH)
    print(f'Imagem blurred: {Y_PATH}    -  Shape: {y.shape} - min: {y.min()}    - max: {y.max()}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH)
        print(f'Imagem limpa: {X_PATH}    -  Shape: {x.shape} - min: {x.min()}  - max: {x.max()}')
    else:
        is_supervised = False
        x = None

    train_ssl_bd(y, x, is_supervised, n_epochs=EPOCHS, learning_rate=LR, base_channels=BASE_CHANNELS, verbose_every=10, results_dir=RESULTS_DIR)

    print('Fim do processamento')