"""
model.py
=========

Reúne os módulos anteriores (wavelet_estimation, networks, losses) para
implementar o algoritmo completo SSL-BD descrito na Seção 2.1 e na
Figura 1 do artigo.

Também implementa a rotação de fase do wavelet inicial de fase zero
(Equação 5):

    w_pred = Re[ (w0 + j * hilbert(w0)) * exp(-j * theta_pred) ]
"""

import torch
import torch.nn as nn

from networks import ReflectivityInversionNet, WaveletPhaseInversionNet
from losses import apply_relative_sparsity, reconstruct_seismic_data, ssl_bd_loss


def hilbert_transform(signal: torch.Tensor) -> torch.Tensor:
    """
    Calcula a transformada de Hilbert de um sinal real 1D usando a FFT,
    equivalente ao que `scipy.signal.hilbert` faz (retorna a parte
    imaginária do sinal analítico), reimplementada em PyTorch para poder
    ser usada dentro do grafo de autograd.

    Parâmetros
    ----------
    signal : torch.Tensor, shape (N,)
        Sinal real de entrada (o wavelet w0).

    Retorna
    -------
    torch.Tensor, shape (N,)
        hilbert(signal): parte imaginária do sinal analítico.
    """
    n = signal.shape[-1]
    spectrum = torch.fft.fft(signal)

    # constrói o filtro do sinal analítico no domínio da frequência
    h = torch.zeros(n, device=signal.device, dtype=spectrum.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    analytic_signal = torch.fft.ifft(spectrum * h)
    return analytic_signal.imag


def rotate_wavelet_phase(w0: torch.Tensor, theta_pred: torch.Tensor) -> torch.Tensor:
    """
    Realiza a rotação de fase do wavelet inicial de fase zero, conforme
    a Equação (5) do artigo:

        w_pred = Re[ (w0 + j*hilbert(w0)) * exp(-j*theta_pred) ]

    Parâmetros
    ----------
    w0 : torch.Tensor, shape (L,)
        Wavelet inicial de fase zero (Eq. 4), fixo durante o treinamento.
    theta_pred : torch.Tensor, escalar
        Fase prevista pela rede de inversão de fase do wavelet.

    Retorna
    -------
    torch.Tensor, shape (L,)
        Wavelet com a fase rotacionada, w_pred.
    """
    w0_hilbert = hilbert_transform(w0)

    # sinal analítico complexo: w0 + j*hilbert(w0)
    analytic = torch.complex(w0, w0_hilbert)

    # rotação de fase: multiplicação por exp(-j*theta)
    rotation = torch.complex(torch.cos(theta_pred), -torch.sin(theta_pred))
    rotated = analytic * rotation

    return rotated.real


class SSLBD(nn.Module):
    """
    Modelo completo de deconvolução cega auto-supervisionada (SSL-BD),
    reunindo os três módulos da Figura 1:

        1. estimativa do wavelet inicial de fase zero (feita fora da
           rede, uma única vez, com `wavelet_estimation.py`);
        2. rede de inversão de fase do wavelet;
        3. rede de inversão de refletividade;

    e implementando o passo de reconstrução + perda auto-supervisionada.
    """

    def __init__(self, w0: torch.Tensor, base_channels: int = 16, ground_truth: torch.Tensor = None, is_supervised: bool = False):
        """
        Parâmetros
        ----------
        w0 : torch.Tensor, shape (L,)
            Wavelet inicial de fase zero, já estimado (Equação 4),
            mantido fixo (buffer, não treinável) durante o treinamento.
        base_channels : int
            Número de canais da primeira camada da rede de refletividade.
        """
        super().__init__()
        self.register_buffer("w0", w0)

        self.reflectivity_net = ReflectivityInversionNet(base_channels=base_channels)
        self.wavelet_phase_net = WaveletPhaseInversionNet()

        self.ground_truth = ground_truth
        self.is_supervised = is_supervised

    def forward(self, y_obs: torch.Tensor, mu: float):
        """
        Executa um passo completo do algoritmo SSL-BD (passos 1-4 da
        Seção 2.1).

        Parâmetros
        ----------
        y_obs : torch.Tensor, shape (batch, 1, n_traces, n_samples)
            Dado sísmico observado Y_obs.
        mu : float
            Parâmetro de esparsidade relativa desta época (Equação 7),
            calculado externamente com `losses.relative_sparsity_mu`.

        Retorna
        -------
        dict com:
            'loss'         : valor escalar da função de perda (Eq. 8)
            'y_pred'       : dado sísmico reconstruído
            'reflectivity' : refletividade prevista (antes da esparsidade)
            'reflectivity_sparse' : refletividade após esparsidade relativa
            'wavelet'      : wavelet com fase rotacionada
            'theta_pred'   : fase prevista (radianos)
        """
        # 1. refletividade prevista pela rede residual
        r_pred = self.reflectivity_net(y_obs)

        # 1. fase prevista pela rede de inversão de fase
        traces = y_obs.squeeze(1)  # shape (batch, n_traces, n_samples)
        theta_pred = self.wavelet_phase_net(traces)
        theta_pred_scalar = theta_pred.mean()  # wavelet estacionário: 1 fase p/ todo o dado

        # 2. rotação de fase do wavelet inicial de fase zero (Eq. 5)
        w_pred = rotate_wavelet_phase(self.w0, theta_pred_scalar)

        # 2. processo de esparsidade relativa sobre a refletividade (Eq. 6-7)
        r_tilde = apply_relative_sparsity(r_pred, mu)

        # 3. reconstrução do dado sísmico: convolução(w_pred, R_tilde)
        y_pred = reconstruct_seismic_data(w_pred, r_tilde)

        # 4. função de perda auto-supervisionada (Eq. 8)
        loss = ssl_bd_loss(y_obs, y_pred)

        if self.is_supervised:
            loss = ssl_bd_loss(self.ground_truth, r_pred)

        return {
            "loss": loss,
            "y_pred": y_pred,
            "reflectivity": r_pred,
            "reflectivity_sparse": r_tilde,
            "wavelet": w_pred,
            "theta_pred": theta_pred_scalar,
        }