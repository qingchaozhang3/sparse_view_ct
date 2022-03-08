#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor laplacian_cuda_forward(torch::Tensor input, int k);

torch::Tensor laplacian_forward(torch::Tensor input, int k) {
  CHECK_INPUT(input);
  return laplacian_cuda_forward(input, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &laplacian_forward, "Graph Laplacian computation forward (CUDA)");
}