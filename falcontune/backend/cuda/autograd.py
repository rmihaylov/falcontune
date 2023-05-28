import torch
from torch.cuda.amp import custom_bwd, custom_fwd

import quant_cuda


# Global Buffer
buffer_mat_dic = {}
cache_buffer = True


def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if not cache_buffer:
        return torch.zeros((shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device)

    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros((shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device)
    else:
        if buffer_mat_dic[shape_of_qweight].device != device:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(device)

        if buffer_mat_dic[shape_of_qweight].dtype != dtype:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(dtype=dtype)

    return buffer_mat_dic[shape_of_qweight]


def matmul4bit_recons(x, qweight, scales, zeros, g_idx, transpose=False):
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons(qweight, buffer, scales, zeros, g_idx)
    output = torch.matmul(x, buffer) if not transpose else torch.matmul(x, buffer.T)
    return output


class AutogradMatmul(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        if bits not in [4]:
            raise NotImplemented('bits in [4]')

        ctx.save_for_backward(qweight, scales, zeros, g_idx, bits)
        output = matmul4bit_recons(x, qweight, scales, zeros, g_idx)
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx, bits = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = matmul4bit_recons(grad_output, qweight, scales, zeros, g_idx, transpose=True)

        return grad_input, None, None, None, None, None, None
