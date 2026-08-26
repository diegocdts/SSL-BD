"""
networks.py
============

Arquiteturas de rede neural descritas nas Seções 2.3 e 2.4 do artigo:

    - ReflectivityInversionNet: rede residual com conexões de salto longas
      (Figura 3), responsável por prever a refletividade R_pred a partir
      do dado sísmico Y_obs.

    - WaveletPhaseInversionNet: rede convolucional totalmente conectada
      (Figura 4), responsável por prever a fase do wavelet theta_pred a
      partir do dado sísmico Y_obs.

Implementação em PyTorch, para o caso 2D (dado sísmico como uma imagem
[traços x amostras no tempo]), conforme indicado no artigo: "For the 2D
case, we can directly replace the 3D convolution layer with a 2D
convolution layer".

IMPORTANTE (fidelidade ao artigo): o artigo afirma explicitamente que a
rede de refletividade "does not have up-sampling or down-sampling
operators [...] so there will be no mismatch in data size during
concatenation". Ou seja, apesar da topologia lembrar uma U-Net (canais
aumentando e depois diminuindo, com conexões de salto longas concatenando
features do "encoder" com o "decoder"), a resolução espacial (número de
amostras) é mantida constante em todas as camadas -- só o número de
canais muda. Por isso todas as convoluções abaixo usam stride=1 e padding
"same".
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Rede de inversão de refletividade (Figura 3)
# ---------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Bloco residual usado ao longo da rede de refletividade (Figura 3):
    duas camadas de convolução, cada uma seguida de LeakyReLU, mais uma
    conexão de atalho (shortcut/"block copied") que soma a entrada à
    saída das duas convoluções -- estrutura clássica de rede residual.

    kernel_size_first permite usar o kernel especial 29x29, exigido pelo
    artigo apenas na primeira convolução do primeiro bloco residual da
    rede ("to capture features on a larger scale").
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size_first: int = 3):
        super().__init__()
        pad_first = kernel_size_first // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size_first,
                                stride=1, padding=pad_first)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1)
        self.act2 = nn.LeakyReLU(inplace=True)

        # projeção 1x1 no atalho apenas quando o número de canais muda,
        # para que a soma residual seja possível
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        return out + identity  # conexão de atalho ("block copied" + Add)


class ReflectivityInversionNet(nn.Module):
    """
    Rede de inversão de refletividade (Figura 3): sequência de blocos
    residuais com número de canais crescente (16, 32, 64, 128, 256) até
    um "gargalo", seguida por blocos residuais com número de canais
    decrescente que recebem, via conexão de salto longa (concatenação),
    as features do caminho crescente correspondente.

    Entrada : dado sísmico pós-empilhamento, 1 canal, shape
              (batch, 1, n_traces, n_samples)
    Saída   : refletividade prevista, mesmo shape da entrada.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_channels, base_channels * 2,
                               base_channels * 4, base_channels * 8,
                               base_channels * 16)

        # --- caminho de canais crescentes ("encoder", sem redução espacial) ---
        # primeira convolução da rede: kernel grande 29x29 (Seção 2.3)
        self.enc1 = ResidualBlock(1, c1, kernel_size_first=29)
        self.enc2 = ResidualBlock(c1, c2, kernel_size_first=3)
        self.enc3 = ResidualBlock(c2, c3, kernel_size_first=3)
        self.enc4 = ResidualBlock(c3, c4, kernel_size_first=3)
        self.bottleneck = ResidualBlock(c4, c5, kernel_size_first=3)

        # --- caminho de canais decrescentes ("decoder"), com concatenação
        #     das conexões de salto longas (long skip connections) ---
        self.dec4 = ResidualBlock(c4 + c5, c4, kernel_size_first=3)
        self.dec3 = ResidualBlock(c3 + c4, c3, kernel_size_first=3)
        self.dec2 = ResidualBlock(c2 + c3, c2, kernel_size_first=3)
        self.dec1 = ResidualBlock(c1 + c2, c1, kernel_size_first=3)

        # última convolução da rede: kernel 1x1x1 para integração entre
        # canais (yellow arrow na Figura 3), reduzindo para 1 canal final
        self.final_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([e4, b], dim=1))
        d3 = self.dec3(torch.cat([e3, d4], dim=1))
        d2 = self.dec2(torch.cat([e2, d3], dim=1))
        d1 = self.dec1(torch.cat([e1, d2], dim=1))

        reflectivity = self.final_conv(d1)
        return reflectivity


# ---------------------------------------------------------------------
# Rede de inversão de fase do wavelet (Figura 4)
# ---------------------------------------------------------------------

class WaveletPhaseInversionNet(nn.Module):
    """
    Rede de inversão de fase do wavelet (Seção 2.4, Figura 4): recebe
    todos os traços do dado sísmico e produz um único valor escalar, a
    fase prevista do wavelet (theta_pred).

    Estrutura (conforme o texto do artigo):
        1. Conv1d(kernel=29, stride=14) + LeakyReLU            -> 16 canais
        2. Conv1d(kernel=4, stride=2) + BatchNorm + LeakyReLU  -> 32 canais
        3. Conv1d(kernel=4, stride=2) + BatchNorm + LeakyReLU  -> 64 canais
        4. Flatten
        5. Fully Connected + LeakyReLU                          -> 256 neurônios
        6. Fully Connected                                      -> 1 (theta_pred)

    Observação de implementação: o comprimento final da sequência após as
    convoluções depende do comprimento L de entrada e não é garantido
    a priori. Para permitir qualquer L (assim como a rede de refletividade
    aceita qualquer tamanho de entrada, conforme o artigo), usamos uma
    camada de pooling adaptativo antes de achatar (flatten) o tensor,
    fixando o comprimento da sequência em L_seq = max(L // 8, 1)
    amostras, coerente com a notação L/8 usada na Figura 4.
    """

    def __init__(self, n_channels_conv: int = 64, hidden_fc: int = 256):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=29, stride=14, padding=14)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv1d(32, n_channels_conv, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(n_channels_conv)
        self.act3 = nn.LeakyReLU(inplace=True)

        # comprimento fixo da sequência de saída das convoluções,
        # equivalente a L/8 no artigo
        self._seq_len = 8
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self._seq_len)

        self.fc1 = nn.Linear(n_channels_conv * self._seq_len, hidden_fc)
        self.act_fc1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_fc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, n_traces, n_samples) -- todos os traços do dado
           sísmico são tratados como um "batch" de traços 1D, conforme o
           artigo ("input of the network consists of all traces of
           seismic data").

        Retorna: escalar theta_pred (fase prevista do wavelet, em
        radianos), obtido pela média das previsões de todos os traços.
        """
        batch, n_traces, n_samples = x.shape
        x = x.reshape(batch * n_traces, 1, n_samples)

        out = self.act1(self.conv1(x))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.act3(self.bn3(self.conv3(out)))

        out = self.adaptive_pool(out)
        out = out.flatten(start_dim=1)

        out = self.act_fc1(self.fc1(out))
        theta_pred = self.fc2(out)  # shape (batch*n_traces, 1)

        # a fase do wavelet é única para o dado (wavelet estacionário),
        # portanto agregamos (média) as previsões de todos os traços
        theta_pred = theta_pred.view(batch, n_traces).mean(dim=1)
        return theta_pred