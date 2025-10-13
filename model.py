# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import constriction
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ops import GDN, LowerBoundFunction
from ops import quantize_ste as quantize


class MeanScaleHyperprior(nn.Module):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        GC        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M):
        super().__init__()
        self.N, self.M = N, M

        self.g_a = nn.Sequential(
            nn.Conv2d(3, N),
            GDN(N),
            nn.Conv2d(N, N),
            GDN(N),
            nn.Conv2d(N, N),
            GDN(N),
            nn.Conv2d(N, M),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3),
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M, N, 5, 2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, 2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, M, 5, 2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(M, M * 3 // 2, 5, 2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 3 // 2, M * 2, 3, padding=1),
        )

        self.z_scales = nn.Parameter(torch.randn(1, N, 1, 1), requires_grad=True)
        self.z_means = nn.Parameter(torch.randn(1, N, 1, 1), requires_grad=True)

        self.lowerbound_scales = 0.11
        self.lowerbound_likelihoods = 1e-9

    def forward(self, x, include_strings=False):
        #######################################################################################
        # Model forward                                                                       #
        #######################################################################################
        outputs = {}

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat = quantize(z - self.z_means) + self.z_means

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = quantize(y - means_hat) + means_hat
        x_hat = self.g_s(y_hat)
        outputs["x_hat"] = x_hat

        z_bpp = self.estimate_bits(z, self.z_scales, self.z_means)
        y_bpp = self.estimate_bits(y, scales_hat, means_hat)
        outputs["estimated_bpps"] = {"y": y_bpp, "z": z_bpp}

        ############## 엔트로피 코딩을 하지 않는 경우 여기에서 return ##############
        if not include_strings:
            return outputs

        #######################################################################################
        # Actual entropy coding happens here                                                  #
        #######################################################################################
        self.gaussian_coder = constriction.stream.stack.AnsCoder()
        self.entropy_model = constriction.stream.model.QuantizedGaussian(-256, 256)

        z_for_code = torch.round(z - self.z_means)
        y_for_code = torch.round(y - means_hat)

        z_scales = torch.clamp(self.z_scales, 0.11)
        scales_hat = torch.clamp(scales_hat, 0.11)

        # round(z)에 대해 N(self.z_means, self.z_scales)으로 엔트로피 코딩하는 대신,
        # round(z - self.z_means)에 대해 N(0, self.z_scales)으로 엔트로피 코딩 수행
        # -> 실험적으로 더 높은 압축 효율; 아래 링크 참고
        # https://groups.google.com/g/tensorflow-compression/c/LQtTAo6l26U/m/cD4ZzmJUAgAJ

        self.gaussian_coder.encode_reverse(
            self.to_1d_numpy(z_for_code, np.int32),
            self.entropy_model,
            self.to_1d_numpy(torch.zeros_like(z), np.float32),
            self.to_1d_numpy(z_scales, np.float32, expand_as=z),
        )
        z_strings = np.array(self.gaussian_coder.get_compressed()).tobytes()

        self.gaussian_coder.clear()

        self.gaussian_coder.encode_reverse(
            self.to_1d_numpy(y_for_code, np.int32),
            self.entropy_model,
            self.to_1d_numpy(torch.zeros_like(y), np.float32),
            self.to_1d_numpy(scales_hat, np.float32),
        )
        y_strings = np.array(self.gaussian_coder.get_compressed()).tobytes()

        outputs["compressed"] = {
            "strings": [[y_strings], [z_strings]],
            "shape": z.shape[-2:],
        }
        return outputs

    def decompress(self, strings, shape):
        device = next(self.parameters()).device
        assert isinstance(strings, list) and len(strings) == 2
        self.entropy_model = constriction.stream.model.QuantizedGaussian(-256, 256)

        # decode z
        self.gaussian_coder = constriction.stream.stack.AnsCoder(
            np.frombuffer(strings[0][1], dtype=np.uint32)
        )
        dummy_z = torch.zeros((self.N, *shape))  # to indicate z's size
        z_scales = torch.clamp(self.z_scales, 0.11)

        # 인코딩 된 round(z - z_means)를 먼저 N(0, self.z_scales)로 디코딩한 후,
        # self.z_means를 더해 z_hat 복원
        z_symbols = self.gaussian_coder.decode(
            self.entropy_model,
            self.to_1d_numpy(dummy_z, np.float32),
            self.to_1d_numpy(z_scales, np.float32, expand_as=dummy_z),
        )
        z_hat = self.to_2d_tensor(z_symbols, dummy_z.shape, device) + self.z_means

        # decode y, 동일한 방식으로 y_hat 복원
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scales_hat = torch.clamp_(scales_hat, 0.11)

        self.gaussian_coder = constriction.stream.stack.AnsCoder(
            np.frombuffer(strings[0][0], dtype=np.uint32)
        )
        y_symbols = self.gaussian_coder.decode(
            self.entropy_model,
            self.to_1d_numpy(torch.zeros_like(means_hat), np.float32),
            self.to_1d_numpy(scales_hat, np.float32),
        )
        y_hat = self.to_2d_tensor(y_symbols, means_hat.shape, device) + means_hat
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def estimate_bits(self, inputs: Tensor, scales: Tensor, means: Tensor) -> Tensor:
        half = float(0.5)

        if self.training:
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
        else:
            inputs = torch.round(inputs - means) + means

        # scale과 likelihood는 0 이상의 값이어야 함, 적당히 작은 값을 각각 최솟값으로 부여
        scales = LowerBoundFunction.apply(scales, self.lowerbound_scales)

        prior = torch.distributions.Normal(means, scales)
        likelihoods_upper = prior.cdf(inputs + half)
        likelihoods_lower = prior.cdf(inputs - half)

        likelihoods = likelihoods_upper - likelihoods_lower
        likelihoods = LowerBoundFunction.apply(likelihoods, self.lowerbound_likelihoods)

        bits = -torch.log2(likelihoods)
        return bits

    @staticmethod
    def to_1d_numpy(
        inputs: Tensor, dtype: np.dtype, expand_as: None | Tensor
    ) -> np.ndarray:
        # Constriction library가 1-D numpy array만을 입력으로 하므로, 그에 맞게 변형
        if expand_as is not None:
            inputs = inputs.expand_as(expand_as)
        return inputs.flatten().cpu().numpy().astype(dtype)

    def to_2d_tensor(self, inputs: np.ndarray, shape: torch.Size, device=torch.device):
        return torch.from_numpy(inputs).to(device=device).reshape(*shape).unsqueeze_(0)
