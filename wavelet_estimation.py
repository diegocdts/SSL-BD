"""
wavelet_estimation.py
======================

Implementa a estimativa do wavelet inicial de fase zero, descrita na
Seção 2.2 e ilustrada na Figura 2 do artigo:

    "Seismic Blind Deconvolution Based on Self-Supervised Machine Learning"
    (Yin, Xu, Yang & Wu, Appl. Sci. 2024, 14, 5214)

Equação (4) do artigo:

    w0 = IFFT( smooth^(3L) ( abs( FFT( (1/Ntr) * sum_i y_i ) ) ) )

onde:
    y_i        : i-ésimo traço do dado sísmico observado
    Ntr        : número de traços
    L          : número de amostras no tempo de cada traço
    smooth^k(.): aplicação repetida (k vezes) de um suavizador polinomial
                 cúbico local de 7 pontos ao espectro de amplitude

Passos (Figura 2):
    (a) traço médio dos dados sísmicos
    (b) espectro de amplitude do traço médio
    (c) espectro de amplitude suavizado
    (d) wavelet de fase zero (IFFT do espectro suavizado)
"""

import numpy as np
from scipy.signal import savgol_filter


def _smooth_amplitude_spectrum(amplitude_spectrum: np.ndarray, n_repeats: int) -> np.ndarray:
    """
    Aplica repetidamente um suavizador polinomial cúbico local com janela
    de 7 pontos ao espectro de amplitude.

    O suavizador polinomial cúbico local de 7 pontos, aplicado a dados
    igualmente espaçados, é exatamente o que a literatura clássica chama
    de filtro de Savitzky-Golay (janela = 7, ordem do polinômio = 3), tal
    como descrito nas referências [24,25] citadas no artigo (Hildebrand,
    1987; Cole & Davie, 1969). Por isso usamos a função já pronta e
    otimizada `scipy.signal.savgol_filter`, em vez de reimplementar o
    ajuste polinomial ponto a ponto.

    Parâmetros
    ----------
    amplitude_spectrum : np.ndarray, shape (L,)
        Espectro de amplitude a ser suavizado.
    n_repeats : int
        Número de vezes que o suavizador de 7 pontos é aplicado
        sucessivamente (no artigo, definido como 3L).

    Retorna
    -------
    np.ndarray, shape (L,)
        Espectro de amplitude suavizado.
    """
    smoothed = amplitude_spectrum.astype(float).copy()
    for _ in range(n_repeats):
        smoothed = savgol_filter(
            smoothed,
            window_length=7,   # suavizador de 7 pontos
            polyorder=3,       # polinômio cúbico local
            mode="nearest",    # trata as bordas do espectro
        )
    return smoothed


def estimate_zero_phase_wavelet(seismic_data: np.ndarray, n_repeats: int = None) -> np.ndarray:
    """
    Estima o wavelet inicial de fase zero w0 a partir dos dados sísmicos
    observados (Equação 4 do artigo).

    Parâmetros
    ----------
    seismic_data : np.ndarray, shape (Ntr, L)
        Dados sísmicos observados Y_obs: Ntr traços, cada um com L
        amostras no tempo.
    n_repeats : int, opcional
        Número de aplicações sucessivas do suavizador de 7 pontos. Se
        None, usa 3*L, exatamente como definido no artigo
        (smooth^(3L)).

    Retorna
    -------
    w0 : np.ndarray, shape (L,)
        Wavelet inicial de fase zero, real, no domínio do tempo, com o
        pico centralizado (fftshift), como mostrado na Figura 2d.
    """
    seismic_data = np.asarray(seismic_data, dtype=float)
    if seismic_data.ndim != 2:
        raise ValueError("seismic_data deve ter shape (Ntr, L)")

    n_tr, length = seismic_data.shape
    if n_repeats is None:
        n_repeats = 3 * length

    # (a) traço médio dos dados sísmicos observados
    mean_trace = seismic_data.mean(axis=0)

    # (b) espectro de amplitude do traço médio
    spectrum = np.fft.fft(mean_trace)
    amplitude_spectrum = np.abs(spectrum)

    # (c) suavização repetida do espectro de amplitude
    smoothed_spectrum = _smooth_amplitude_spectrum(amplitude_spectrum, n_repeats)

    # (d) IFFT do espectro suavizado -> wavelet de fase zero no tempo
    w0_complex = np.fft.ifft(smoothed_spectrum)
    w0 = np.real(w0_complex)

    # centraliza o pico do wavelet (equivalente ao que é mostrado nos
    # gráficos do artigo, com o wavelet centrado em t=0)
    w0 = np.fft.fftshift(w0)

    return w0