from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graph_laplacian_2',
    ext_modules=[
        CUDAExtension('graph_laplacian_2', [
            'laplacian_cuda.cpp',
            'laplacian_cuda_kernel.cu',
        ]#,
        #library_dirs=['C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64'],
        #extra_link_args=['c10_cuda.lib','cublas.lib']
        )
    ],    
    cmdclass={
        'build_ext': BuildExtension
    })
