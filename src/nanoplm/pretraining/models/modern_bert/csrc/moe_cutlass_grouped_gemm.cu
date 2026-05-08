#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/half.h>

#include <cuda_runtime.h>

namespace {

constexpr int kAlignment = 8;

template <typename Element>
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element,
    128 / cutlass::sizeof_bits<Element>::value,
    float,
    float>;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using GroupedGemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    kAlignment,
    Element,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    kAlignment,
    Element,
    LayoutC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp<Element>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using GroupedGemm = cutlass::gemm::device::GemmGrouped<
    GroupedGemmKernel<Element, LayoutA, LayoutB, LayoutC>>;

template <typename Element>
__global__ void build_grouped_metadata_kernel(
    const Element* a,
    const Element* b,
    Element* c,
    const int64_t* batch_sizes,
    int32_t groups,
    int32_t a_cols,
    int32_t b_dim1,
    int32_t b_dim2,
    bool trans_a,
    bool trans_b,
    cutlass::gemm::GemmCoord* problem_sizes,
    int64_t* ptr_a,
    int64_t* ptr_b,
    int64_t* ptr_c,
    int64_t* ptr_d,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    int64_t* ldd) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t row_offset = 0;
  int64_t out_offset = 0;

  for (int32_t group_idx = 0; group_idx < groups; ++group_idx) {
    int32_t rows = static_cast<int32_t>(batch_sizes[group_idx]);

    if (trans_a) {
      problem_sizes[group_idx] = cutlass::gemm::GemmCoord(a_cols, b_dim1, rows);
      ptr_a[group_idx] = reinterpret_cast<int64_t>(a + row_offset * a_cols);
      ptr_b[group_idx] = reinterpret_cast<int64_t>(b + row_offset * b_dim1);
      ptr_c[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      ptr_d[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      lda[group_idx] = a_cols;
      ldb[group_idx] = b_dim1;
      ldc[group_idx] = b_dim1;
      ldd[group_idx] = b_dim1;
      out_offset += static_cast<int64_t>(a_cols) * b_dim1;
    } else if (trans_b) {
      problem_sizes[group_idx] = cutlass::gemm::GemmCoord(rows, b_dim1, a_cols);
      ptr_a[group_idx] = reinterpret_cast<int64_t>(a + row_offset * a_cols);
      ptr_b[group_idx] = reinterpret_cast<int64_t>(
          b + static_cast<int64_t>(group_idx) * b_dim1 * b_dim2);
      ptr_c[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      ptr_d[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      lda[group_idx] = a_cols;
      ldb[group_idx] = b_dim2;
      ldc[group_idx] = b_dim1;
      ldd[group_idx] = b_dim1;
      out_offset += static_cast<int64_t>(rows) * b_dim1;
    } else {
      problem_sizes[group_idx] = cutlass::gemm::GemmCoord(rows, b_dim2, a_cols);
      ptr_a[group_idx] = reinterpret_cast<int64_t>(a + row_offset * a_cols);
      ptr_b[group_idx] = reinterpret_cast<int64_t>(
          b + static_cast<int64_t>(group_idx) * a_cols * b_dim2);
      ptr_c[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      ptr_d[group_idx] = reinterpret_cast<int64_t>(c + out_offset);
      lda[group_idx] = a_cols;
      ldb[group_idx] = b_dim2;
      ldc[group_idx] = b_dim2;
      ldd[group_idx] = b_dim2;
      out_offset += static_cast<int64_t>(rows) * b_dim2;
    }

    row_offset += rows;
  }
}

void check_inputs(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b) {
  TORCH_CHECK(a.is_cuda(), "grouped_gemm expects CUDA lhs tensor");
  TORCH_CHECK(b.is_cuda(), "grouped_gemm expects CUDA rhs tensor");
  TORCH_CHECK(batch_sizes.is_cuda(), "grouped_gemm expects CUDA batch_sizes tensor");
  TORCH_CHECK(a.dim() == 2, "grouped_gemm expects a 2D lhs tensor");
  TORCH_CHECK(batch_sizes.dim() == 1, "grouped_gemm expects a 1D batch_sizes tensor");
  TORCH_CHECK(
      batch_sizes.scalar_type() == torch::kInt64,
      "grouped_gemm expects int64 batch_sizes tensor");
  TORCH_CHECK(!(trans_a && trans_b), "grouped_gemm does not support trans_a and trans_b together");
  if (trans_a) {
    TORCH_CHECK(b.dim() == 2, "grouped_gemm expects a 2D rhs tensor when trans_a=True");
  } else {
    TORCH_CHECK(b.dim() == 3, "grouped_gemm expects a 3D rhs tensor when trans_a=False");
  }
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "grouped_gemm expects matching input dtypes");
  TORCH_CHECK(
      a.scalar_type() == torch::kFloat16 || a.scalar_type() == torch::kBFloat16,
      "CUTLASS grouped_gemm currently supports float16 and bfloat16 only");
}

torch::Tensor allocate_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b) {
  if (trans_a) {
    return torch::empty(
        {batch_sizes.size(0), a.size(1), b.size(1)},
        a.options().memory_format(torch::MemoryFormat::Contiguous));
  }
  if (trans_b) {
    return torch::empty(
        {a.size(0), b.size(1)},
        a.options().memory_format(torch::MemoryFormat::Contiguous));
  }
  return torch::empty(
      {a.size(0), b.size(2)},
      a.options().memory_format(torch::MemoryFormat::Contiguous));
}

template <typename Element>
void build_metadata(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b,
    torch::Tensor& problem_sizes,
    torch::Tensor& ptr_a,
    torch::Tensor& ptr_b,
    torch::Tensor& ptr_c,
    torch::Tensor& ptr_d,
    torch::Tensor& lda,
    torch::Tensor& ldb,
    torch::Tensor& ldc,
    torch::Tensor& ldd,
    cudaStream_t stream) {
  int32_t groups = static_cast<int32_t>(batch_sizes.size(0));
  int32_t a_cols = static_cast<int32_t>(a.size(1));
  int32_t b_dim1 = static_cast<int32_t>(b.size(1));
  int32_t b_dim2 = static_cast<int32_t>(b.dim() == 3 ? b.size(2) : 0);

  build_grouped_metadata_kernel<Element><<<1, 1, 0, stream>>>(
      reinterpret_cast<const Element*>(a.data_ptr()),
      reinterpret_cast<const Element*>(b.data_ptr()),
      reinterpret_cast<Element*>(c.data_ptr()),
      batch_sizes.data_ptr<int64_t>(),
      groups,
      a_cols,
      b_dim1,
      b_dim2,
      trans_a,
      trans_b,
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes.data_ptr<int32_t>()),
      ptr_a.data_ptr<int64_t>(),
      ptr_b.data_ptr<int64_t>(),
      ptr_c.data_ptr<int64_t>(),
      ptr_d.data_ptr<int64_t>(),
      lda.data_ptr<int64_t>(),
      ldb.data_ptr<int64_t>(),
      ldc.data_ptr<int64_t>(),
      ldd.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Gemm>
void run_cutlass_grouped_gemm(
    torch::Tensor& problem_sizes,
    torch::Tensor& ptr_a,
    torch::Tensor& ptr_b,
    torch::Tensor& ptr_c,
    torch::Tensor& ptr_d,
    torch::Tensor& lda,
    torch::Tensor& ldb,
    torch::Tensor& ldc,
    torch::Tensor& ldd,
    int32_t groups,
    cudaStream_t stream) {
  using Element = typename Gemm::ElementA;
  typename Gemm::EpilogueOutputOp::Params output_op(1.0f, 0.0f);
  typename Gemm::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes.data_ptr<int32_t>()),
      groups,
      Gemm::sufficient(nullptr, 0),
      output_op,
      reinterpret_cast<Element**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_c.data_ptr<int64_t>()),
      reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
      lda.data_ptr<int64_t>(),
      ldb.data_ptr<int64_t>(),
      ldc.data_ptr<int64_t>(),
      ldd.data_ptr<int64_t>(),
      nullptr);

  Gemm gemm;
  cutlass::Status status = gemm.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS grouped_gemm can_implement failed with status ",
      cutlassGetStatusString(status));

  status = gemm.initialize(args, nullptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS grouped_gemm initialize failed with status ",
      cutlassGetStatusString(status));

  status = gemm.run(stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS grouped_gemm launch failed with status ",
      cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Element>
torch::Tensor grouped_gemm_cuda_impl(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b) {
  static_assert(sizeof(cutlass::gemm::GemmCoord) == sizeof(int32_t) * 3);

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto batch_sizes_contig = batch_sizes.contiguous();
  auto c = allocate_output(a_contig, b_contig, batch_sizes_contig, trans_a, trans_b);

  auto meta_options = torch::TensorOptions()
                          .device(a_contig.device())
                          .dtype(torch::kInt64);
  auto problem_options = torch::TensorOptions()
                             .device(a_contig.device())
                             .dtype(torch::kInt32);
  int64_t groups = batch_sizes_contig.size(0);

  auto problem_sizes = torch::empty({groups, 3}, problem_options);
  auto ptr_a = torch::empty({groups}, meta_options);
  auto ptr_b = torch::empty({groups}, meta_options);
  auto ptr_c = torch::empty({groups}, meta_options);
  auto ptr_d = torch::empty({groups}, meta_options);
  auto lda = torch::empty({groups}, meta_options);
  auto ldb = torch::empty({groups}, meta_options);
  auto ldc = torch::empty({groups}, meta_options);
  auto ldd = torch::empty({groups}, meta_options);

  auto stream = at::cuda::getCurrentCUDAStream(a_contig.device().index()).stream();
  build_metadata<Element>(
      a_contig,
      b_contig,
      c,
      batch_sizes_contig,
      trans_a,
      trans_b,
      problem_sizes,
      ptr_a,
      ptr_b,
      ptr_c,
      ptr_d,
      lda,
      ldb,
      ldc,
      ldd,
      stream);

  if (trans_a) {
    using Gemm = GroupedGemm<
        Element,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor>;
    run_cutlass_grouped_gemm<Gemm>(
        problem_sizes, ptr_a, ptr_b, ptr_c, ptr_d, lda, ldb, ldc, ldd, groups, stream);
  } else if (trans_b) {
    using Gemm = GroupedGemm<
        Element,
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::layout::RowMajor>;
    run_cutlass_grouped_gemm<Gemm>(
        problem_sizes, ptr_a, ptr_b, ptr_c, ptr_d, lda, ldb, ldc, ldd, groups, stream);
  } else {
    using Gemm = GroupedGemm<
        Element,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor,
        cutlass::layout::RowMajor>;
    run_cutlass_grouped_gemm<Gemm>(
        problem_sizes, ptr_a, ptr_b, ptr_c, ptr_d, lda, ldb, ldc, ldd, groups, stream);
  }

  return c;
}

}  // namespace

torch::Tensor grouped_gemm_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b) {
  check_inputs(a, b, batch_sizes, trans_a, trans_b);
  c10::cuda::CUDAGuard device_guard(a.device());

  if (a.scalar_type() == torch::kFloat16) {
    return grouped_gemm_cuda_impl<cutlass::half_t>(a, b, batch_sizes, trans_a, trans_b);
  }
  if (a.scalar_type() == torch::kBFloat16) {
    return grouped_gemm_cuda_impl<cutlass::bfloat16_t>(a, b, batch_sizes, trans_a, trans_b);
  }

  TORCH_CHECK(false, "Unsupported dtype for CUTLASS grouped_gemm: ", a.scalar_type());
}
