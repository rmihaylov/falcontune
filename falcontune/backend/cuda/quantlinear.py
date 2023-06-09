import torch

import quant_cuda
from falcontune.backend.base import QuantLinearBase
from falcontune.backend.cuda.autograd import AutogradMatmul


class QuantLinear(QuantLinearBase):
    framework = 'cuda'

    def forward(self, x):
        if torch.is_grad_enabled():
            out = AutogradMatmul.apply(
                x, self.qweight, self.scales,
                self.qzeros, self.g_idx, self.bits, self.maxq)
        else:
            out_shape = x.shape[:-1] + (self.outfeatures,)
            x = x.reshape(-1, x.shape[-1])

            out = torch.zeros((x.shape[0], self.outfeatures), device=x.device, dtype=torch.float32)

            if self.bits == 2:
                quant_cuda.vecquant2matmul(x.float(), self.qweight, out, self.scales.float(), self.qzeros, self.g_idx)
            elif self.bits == 3:
                quant_cuda.vecquant3matmul(x.float(), self.qweight, out, self.scales.float(), self.qzeros, self.g_idx)
            elif self.bits == 4:
                quant_cuda.vecquant4matmul(x.float(), self.qweight, out, self.scales.float(), self.qzeros, self.g_idx)
            elif self.bits == 8:
                quant_cuda.vecquant8matmul(x.float(), self.qweight, out, self.scales.float(), self.qzeros, self.g_idx)
            else:
                raise NotImplemented('bits in [2, 3, 4, 8]')

            out = out.half()
            out = out.reshape(out_shape)

        if self.bias is not None:
            out += self.bias

        return out
