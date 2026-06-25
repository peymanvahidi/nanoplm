#include <torch/extension.h>

torch::Tensor grouped_gemm_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "grouped_gemm",
      &grouped_gemm_cuda,
      py::arg("a"),
      py::arg("b"),
      py::arg("batch_sizes"),
      py::arg("trans_a") = false,
      py::arg("trans_b") = false);
}
