import torch
from torch.cuda.amp import custom_bwd, custom_fwd

import falcontune.backend.triton.triton_utils as tu


class AutogradMatmul(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, qzeros, g_idx, bits, maxq):
        output = tu.triton_matmul(x, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = tu.triton_matmul_transpose(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)

        return grad_input, None, None, None, None, None, None
