import torch

from falcontune.backend.base import QuantLinearBase


class QuantLinear(QuantLinearBase):
    framework = 'torch'

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])

        if self.bits in [2, 4, 8]:
            zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
                                              self.wf.unsqueeze(0)).to(
                torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                                               self.wf.unsqueeze(-1)).to(
                torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2 ** self.bits) - 1, out=weight)
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1,
                                                                                                      12)
            zeros = (zeros >> self.wf.unsqueeze(0))
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = torch.cat([zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2)

            zeros = zeros + 1
            zeros = zeros.reshape(self.scales.shape)

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1,
                                                                                                          12, -1)
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        else:
            raise NotImplemented('bits in [2, 3, 4, 8]')

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // x.shape[-1]

        if num_itr == 1:
            weights = (self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()]))
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim:(i + 1) * num_dim]
                weight_i = weight[:, i * num_dim:(i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim:(i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim:(i + 1) * num_dim]
                weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights, dim=1)

        out = torch.matmul(x.half(), weights)

        out = out.reshape(out_shape)
        out = (out + self.bias) if (self.bias is not None) else out
        return out
