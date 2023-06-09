import torch

from falcontune.backend.base import QuantLinearBase
import falcontune.backend.triton.triton_utils as tu
from falcontune.backend.triton.autograd import AutogradMatmul


class QuantLinear(QuantLinearBase):
    framework = 'triton'

    def forward(self, x):
        if torch.is_grad_enabled():
            out = AutogradMatmul.apply(
                x, self.qweight, self.scales,
                self.qzeros, self.g_idx, self.bits, self.maxq)
        else:
            assert self.qzeros.dtype == torch.int32
            out = tu.triton_matmul(x, self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)

        if self.bias is not None:
            out += self.bias

        return out
