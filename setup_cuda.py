from setuptools import setup
from torch.utils import cpp_extension


setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda',
        [
            'falcontune/backend/cuda/quant_cuda.cpp',
            'falcontune/backend/cuda/quant_cuda_kernel.cu'
        ]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
