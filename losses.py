"""
losses.py
==========

Implementa o que é descrito na Seção 2.5 do artigo:

    - relative_sparsity_mu(r_epoch): calcula o parâmetro mu conforme a
      Equação (7), que cresce de 0.01 a 0.1 ao longo do treinamento.

    - apply_relative_sparsity(R_pred, mu): aplica o processo de
      esparsidade relativa da Equação (6), zerando amostras da
      refletividade prevista cujo valor absoluto é menor que
      max(|R_pred|) * mu.

    - reconstruct_seismic_data(wavelet, reflectivity): realiza a
      convolução (Equação 1/2) entre o wavelet estacionário e a
      refletividade prevista, trace a trace, para gerar o dado sísmico
      reconstruído Y_pred.

    - ssl_bd_loss(...): função de perda final (Equação 8), o erro
      quadrático entre o dado observado e o dado reconstruído.
"""

import torch
import torch.nn.functional as F


def relative_sparsity_mu(r_epoch: float) -> float:
    """
    Calcula o parâmetro de esparsidade relativa mu, conforme a
    Equação (7) do artigo:

        mu = 0.01                                            se r_epoch < 0.1
        mu = (r_epoch-0.1)/(0.9-0.1)*0.1 + (0.9-r_epoch)/(0.9-0.1)*0.01
                                                              se 0.1 <= r_epoch <= 0.9
        mu = 0.1                                             se r_epoch > 0.9

    Parâmetros
    ----------
    r_epoch : float
        Razão entre a época atual de treinamento e o número total de
        épocas (valor entre 0 e 1).

    Retorna
    -------
    float
        Valor de mu.
    """
    if r_epoch < 0.1:
        return 0.01
    elif r_epoch > 0.9:
        return 0.1
    else:
        return ((r_epoch - 0.1) / (0.9 - 0.1)) * 0.1 + ((0.9 - r_epoch) / (0.9 - 0.1)) * 0.01


def apply_relative_sparsity(R_pred: torch.Tensor, mu: float) -> torch.Tensor:
    """
    Aplica o processo de esparsidade relativa definido na Equação (6):

        R_tilde = R_pred,  se |R_pred| >= max(|R_pred|) * mu
        R_tilde = 0,       se |R_pred| <  max(|R_pred|) * mu

    O máximo é calculado sobre toda a refletividade prevista (por
    amostra do batch), reproduzindo max(abs(R_pred)) do artigo.

    Parâmetros
    ----------
    R_pred : torch.Tensor, shape (batch, 1, n_traces, n_samples)
        Refletividade prevista pela rede.
    mu : float
        Limiar relativo de esparsidade (Equação 7).

    Retorna
    -------
    torch.Tensor, mesmo shape de R_pred
        Refletividade após o processo de esparsidade relativa.
    """
    abs_r = R_pred.abs()
    # máximo por amostra do batch (mantendo dimensões para broadcasting)
    max_abs_r = abs_r.amax(dim=(1, 2, 3), keepdim=True)
    threshold = max_abs_r * mu

    mask = (abs_r >= threshold).float()
    R_tilde = R_pred * mask
    return R_tilde


def reconstruct_seismic_data(wavelet: torch.Tensor, reflectivity: torch.Tensor) -> torch.Tensor:
    """
    Reconstrói o dado sísmico convolvendo o wavelet estacionário (Toeplitz
    convolution, Equações 1 e 2 do artigo) com a refletividade prevista,
    trace a trace, ao longo do eixo do tempo.

    Parâmetros
    ----------
    wavelet : torch.Tensor, shape (wavelet_length,)
        Wavelet sísmico (já rotacionado em fase), assumido estacionário
        (o mesmo para todos os traços).
    reflectivity : torch.Tensor, shape (batch, 1, n_traces, n_samples)
        Refletividade (após o processo de esparsidade relativa).

    Retorna
    -------
    torch.Tensor, mesmo shape de reflectivity
        Dado sísmico reconstruído Y_pred = w_pred * R_pred (convolução).
    """
    batch, _, n_traces, n_samples = reflectivity.shape

    # kernel de convolução 1D no eixo do tempo, aplicado com o mesmo
    # wavelet para todos os traços (peso compartilhado -> estacionário)
    kernel = wavelet.flip(0).view(1, 1, -1)  # flip: convolução real (não correlação)
    pad = wavelet.shape[0] // 2

    # reorganiza para (batch * n_traces, 1, n_samples) e aplica conv1d
    x = reflectivity.reshape(batch * n_traces, 1, n_samples)
    y = F.conv1d(x, kernel, padding=pad)

    # garante mesmo comprimento de saída que a entrada (recorte caso o
    # kernel tenha comprimento par e a convolução gere 1 amostra extra)
    y = y[..., :n_samples]

    y_pred = y.view(batch, 1, n_traces, n_samples)
    return y_pred


def ssl_bd_loss(y_obs: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Função de perda auto-supervisionada (Equação 8 do artigo):

        loss(Y_obs, Y_pred) = || Y_obs - Y_pred ||_2^2

    Implementada como o erro quadrático médio (MSE) entre o dado
    observado e o dado reconstruído, usando a função pronta do PyTorch.
    """
    return F.mse_loss(y_pred, y_obs, reduction="mean")