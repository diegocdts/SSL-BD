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

from wavelet_estimation import estimate_zero_phase_wavelet
from losses import relative_sparsity_mu
from model import SSLBD
from vizualization import plot_images


def train_ssl_bd(
    seismic_data: np.ndarray,
    n_epochs: int = 10000,
    learning_rate: float = 1e-5,
    base_channels: int = 16,
    device: str = None,
    verbose_every: int = 500,
    model_path: str = None
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
    model_path: str
        Caminho para salvar o modelo

    Retorna
    -------
    dict com:
        'model'        : o modelo SSLBD treinado
        'wavelet'      : wavelet final estimado (np.ndarray)
        'reflectivity' : refletividade final prevista (np.ndarray),
                          já com o processo de esparsidade relativa
        'loss_history' : lista com o valor da perda a cada época
    """

    assert model_path is not None, "O path para o modelo precisa ser definido"

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

    # --------------------------------------------------------------
    # Modelo e otimizador (Adam, lr=1e-5, conforme Seção 3)
    # --------------------------------------------------------------
    model = SSLBD(w0=w0, base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kernel_size": kernel_size,
            "num_layers": num_layers,
        },
        model_path
    )

    return {
        "model": model,
        "wavelet": final_wavelet,
        "reflectivity": final_reflectivity,
        "loss_history": loss_history,
    }

def normalize_max_abs(data):
    """
    Normalização global pela maior amplitude absoluta.

    Exemplo:
        data = [-10, -5, 2, 8]
        -> [-1.0, -0.5, 0.2, 0.8]

    Preserva as relações de amplitude entre os traços.
    """

    scale = np.max(np.abs(data))

    if scale == 0:
        return data.copy(), scale

    data_norm = data / scale

    return data_norm


def normalize_minmax(data):
    """
    Normalização min-max global para o intervalo [-1, 1].

    x_norm = 2 * (x - xmin) / (xmax - xmin) - 1
    """

    xmin = np.min(data)
    xmax = np.max(data)

    if xmax == xmin:
        return np.zeros_like(data), xmin, xmax

    data_norm = 2.0 * (data - xmin) / (xmax - xmin) - 1.0

    return data_norm

def load_data(data_path):
    data = np.load(data_path).astype("float32")[0]
    data = data.reshape(data.shape[-2], data.shape[-1])
    data = normalize_max_abs(data)

    return data



if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configurações
    # ------------------------------------------------------------------
    Y_PATH = "/home/data/IN.npy"
    X_PATH = "/home/data/RFLT.npy"
    EPOCHS = 3
    LR = 1e-5
    BASE_CHANNELS = 16
    RESULTS_DIR = f'/home/src/results/SSLBD_{EPOCHS}_{LR}_{BASE_CHANNELS}'
    MODEL_PATH = f'{RESULTS_DIR}/model.pth'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y = load_data(data_path=Y_PATH)
    print(f'Imagem blurred: {y.shape}', f'min: {y.min()} - max: {y.max()}')

    if X_PATH is not None:
        x = load_data(data_path=X_PATH)
        print(f'Imagem limpa:   {x.shape}', f'min: {x.min()} - max: {x.max()}')
    else:
        x = None

    result = train_ssl_bd(y, n_epochs=EPOCHS, learning_rate=LR, base_channels=BASE_CHANNELS, verbose_every=10, model_path=MODEL_PATH)

    reflectivity = result["reflectivity"].detach().cpu().numpy()
    wavelet = result["wavelet"].detach().cpu().numpy()
    loss_history = result["loss_history"]

    np.save(f'{RESULTS_DIR}/reflectivity.npy', reflectivity)
    np.save(f'{RESULTS_DIR}/wavelet.npy', wavelet)
    np.save(f'{RESULTS_DIR}/loss_history.npy', loss_history)

    print("Wavelet final shape:", result["wavelet"].shape)
    print("Refletividade final shape:", result["reflectivity"].shape)

    plot_images(y=y, x_hat=reflectivity, x=x, cmap='seismic', results_dir=RESULTS_DIR, name='final')