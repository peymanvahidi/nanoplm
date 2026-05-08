#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/half.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <string>

namespace {

constexpr int kAlignment = 8;

enum class BackendPreference {
  kAuto,
  kSm80,
  kSm90,
};

BackendPreference backend_preference_from_env() {
  const char* raw = std::getenv("NANOPLM_MOE_GROUPED_GEMM_BACKEND");
  std::string value = raw == nullptr ? "auto" : std::string(raw);
  if (value.empty() || value == "auto") {
    return BackendPreference::kAuto;
  }
  if (value == "sm80") {
    return BackendPreference::kSm80;
  }
  if (value == "sm90") {
    return BackendPreference::kSm90;
  }
  TORCH_CHECK(
      false,
      "Invalid NANOPLM_MOE_GROUPED_GEMM_BACKEND='",
      value,
      "'. Expected one of: auto, sm80, sm90.");
}

bool device_supports_sm90(int device_index) {
  cudaDeviceProp prop;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
  return prop.major >= 9;
}

std::string env_or_default(const char* name, const char* default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return std::string(default_value);
  }
  return std::string(raw);
}

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

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

using HopperProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

template <
    typename Element,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename TileShape_,
    typename ClusterShape_,
    typename KernelSchedule_,
    typename EpilogueSchedule_>
struct HopperGroupedGemmConfig {
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  using ElementAccumulator = float;
  using KernelSchedule = KernelSchedule_;
  using EpilogueSchedule = EpilogueSchedule_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      Element,
      LayoutC*,
      kAlignment,
      Element,
      LayoutC*,
      kAlignment,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<Element, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      Element,
      LayoutA*,
      kAlignment,
      Element,
      LayoutB*,
      kAlignment,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      HopperProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmWide = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmSmallM = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_64, cute::_256, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmTall = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmTallPingpong = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_256, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmWidePingpong = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_128, cute::_256, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmSquarePingpong = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::Gemm;

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
using HopperGroupedGemmSquareCooperative = typename HopperGroupedGemmConfig<
    Element,
    LayoutA,
    LayoutB,
    LayoutC,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::Gemm;

#endif

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

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <
    typename Element,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename StrideD>
__global__ void build_hopper_grouped_metadata_kernel(
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
    typename HopperProblemShape::UnderlyingProblemShape* problem_sizes,
    const Element** ptr_a,
    const Element** ptr_b,
    const Element** ptr_c,
    Element** ptr_d,
    StrideA* stride_a,
    StrideB* stride_b,
    StrideC* stride_c,
    StrideD* stride_d) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t row_offset = 0;
  int64_t out_offset = 0;

  for (int32_t group_idx = 0; group_idx < groups; ++group_idx) {
    int32_t rows = static_cast<int32_t>(batch_sizes[group_idx]);

    if (trans_a) {
      problem_sizes[group_idx] = cute::make_tuple(a_cols, b_dim1, rows);
      ptr_a[group_idx] = a + row_offset * a_cols;
      ptr_b[group_idx] = b + row_offset * b_dim1;
      ptr_c[group_idx] = c + out_offset;
      ptr_d[group_idx] = c + out_offset;
      stride_a[group_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(a_cols, rows, 1));
      stride_b[group_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(b_dim1, rows, 1));
      stride_c[group_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(a_cols, b_dim1, 1));
      stride_d[group_idx] = cutlass::make_cute_packed_stride(
          StrideD{}, cute::make_shape(a_cols, b_dim1, 1));
      out_offset += static_cast<int64_t>(a_cols) * b_dim1;
    } else if (trans_b) {
      problem_sizes[group_idx] = cute::make_tuple(rows, b_dim1, a_cols);
      ptr_a[group_idx] = a + row_offset * a_cols;
      ptr_b[group_idx] = b + static_cast<int64_t>(group_idx) * b_dim1 * b_dim2;
      ptr_c[group_idx] = c + out_offset;
      ptr_d[group_idx] = c + out_offset;
      stride_a[group_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(rows, a_cols, 1));
      stride_b[group_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(b_dim1, a_cols, 1));
      stride_c[group_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(rows, b_dim1, 1));
      stride_d[group_idx] = cutlass::make_cute_packed_stride(
          StrideD{}, cute::make_shape(rows, b_dim1, 1));
      out_offset += static_cast<int64_t>(rows) * b_dim1;
    } else {
      problem_sizes[group_idx] = cute::make_tuple(rows, b_dim2, a_cols);
      ptr_a[group_idx] = a + row_offset * a_cols;
      ptr_b[group_idx] = b + static_cast<int64_t>(group_idx) * a_cols * b_dim2;
      ptr_c[group_idx] = c + out_offset;
      ptr_d[group_idx] = c + out_offset;
      stride_a[group_idx] = cutlass::make_cute_packed_stride(
          StrideA{}, cute::make_shape(rows, a_cols, 1));
      stride_b[group_idx] = cutlass::make_cute_packed_stride(
          StrideB{}, cute::make_shape(b_dim2, a_cols, 1));
      stride_c[group_idx] = cutlass::make_cute_packed_stride(
          StrideC{}, cute::make_shape(rows, b_dim2, 1));
      stride_d[group_idx] = cutlass::make_cute_packed_stride(
          StrideD{}, cute::make_shape(rows, b_dim2, 1));
      out_offset += static_cast<int64_t>(rows) * b_dim2;
    }

    row_offset += rows;
  }
}

#endif

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

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename Stride>
torch::Tensor allocate_stride_tensor(int64_t groups, const torch::TensorOptions& options) {
  static_assert(sizeof(Stride) % sizeof(int64_t) == 0);
  return torch::empty({groups, static_cast<int64_t>(sizeof(Stride) / sizeof(int64_t))}, options);
}

template <typename Gemm>
cutlass::KernelHardwareInfo hopper_hardware_info(int device_index) {
  constexpr int kMaxCachedDevices = 16;
  static std::array<cutlass::KernelHardwareInfo, kMaxCachedDevices> cached = {};
  if (device_index >= 0 && device_index < kMaxCachedDevices) {
    auto& info = cached[device_index];
    if (info.sm_count == 0 || info.max_active_clusters == 0) {
      info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
          typename Gemm::GemmKernel>(device_index);
    }
    return info;
  }
  return cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(
      device_index);
}

template <typename Gemm, typename Element>
void build_hopper_metadata(
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
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_c,
    torch::Tensor& stride_d,
    cudaStream_t stream) {
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  static_assert(sizeof(typename HopperProblemShape::UnderlyingProblemShape) == sizeof(int32_t) * 3);

  int32_t groups = static_cast<int32_t>(batch_sizes.size(0));
  int32_t a_cols = static_cast<int32_t>(a.size(1));
  int32_t b_dim1 = static_cast<int32_t>(b.size(1));
  int32_t b_dim2 = static_cast<int32_t>(b.dim() == 3 ? b.size(2) : 0);

  build_hopper_grouped_metadata_kernel<Element, StrideA, StrideB, StrideC, StrideD>
      <<<1, 1, 0, stream>>>(
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
          reinterpret_cast<typename HopperProblemShape::UnderlyingProblemShape*>(
              problem_sizes.data_ptr<int32_t>()),
          reinterpret_cast<const Element**>(ptr_a.data_ptr<int64_t>()),
          reinterpret_cast<const Element**>(ptr_b.data_ptr<int64_t>()),
          reinterpret_cast<const Element**>(ptr_c.data_ptr<int64_t>()),
          reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
          reinterpret_cast<StrideA*>(stride_a.data_ptr<int64_t>()),
          reinterpret_cast<StrideB*>(stride_b.data_ptr<int64_t>()),
          reinterpret_cast<StrideC*>(stride_c.data_ptr<int64_t>()),
          reinterpret_cast<StrideD*>(stride_d.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Gemm>
void run_hopper_grouped_gemm(
    torch::Tensor& problem_sizes,
    torch::Tensor& ptr_a,
    torch::Tensor& ptr_b,
    torch::Tensor& ptr_c,
    torch::Tensor& ptr_d,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_c,
    torch::Tensor& stride_d,
    int32_t groups,
    int device_index,
    cudaStream_t stream) {
  using Element = typename Gemm::ElementA;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  cutlass::KernelHardwareInfo hw_info = hopper_hardware_info<Gemm>(device_index);

  typename Gemm::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  args = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups,
       reinterpret_cast<typename HopperProblemShape::UnderlyingProblemShape*>(
           problem_sizes.data_ptr<int32_t>()),
       nullptr},
      {reinterpret_cast<const Element**>(ptr_a.data_ptr<int64_t>()),
       reinterpret_cast<StrideA*>(stride_a.data_ptr<int64_t>()),
       reinterpret_cast<const Element**>(ptr_b.data_ptr<int64_t>()),
       reinterpret_cast<StrideB*>(stride_b.data_ptr<int64_t>())},
      {fusion_args,
       reinterpret_cast<const Element**>(ptr_c.data_ptr<int64_t>()),
       reinterpret_cast<StrideC*>(stride_c.data_ptr<int64_t>()),
       reinterpret_cast<Element**>(ptr_d.data_ptr<int64_t>()),
       reinterpret_cast<StrideD*>(stride_d.data_ptr<int64_t>())},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().device(problem_sizes.device()).dtype(torch::kUInt8));
  void* workspace_ptr = workspace_size == 0 ? nullptr : workspace.data_ptr();

  cutlass::Status status = gemm.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS SM90 grouped_gemm can_implement failed with status ",
      cutlassGetStatusString(status));

  status = gemm.initialize(args, workspace_ptr, stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS SM90 grouped_gemm initialize failed with status ",
      cutlassGetStatusString(status));

  status = gemm.run(stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS SM90 grouped_gemm launch failed with status ",
      cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Gemm, typename Element>
void run_hopper_backend(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b,
    int device_index,
    cudaStream_t stream) {
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  auto meta_options = torch::TensorOptions().device(a.device()).dtype(torch::kInt64);
  auto problem_options = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
  int64_t groups = batch_sizes.size(0);

  auto problem_sizes = torch::empty({groups, 3}, problem_options);
  auto ptr_a = torch::empty({groups}, meta_options);
  auto ptr_b = torch::empty({groups}, meta_options);
  auto ptr_c = torch::empty({groups}, meta_options);
  auto ptr_d = torch::empty({groups}, meta_options);
  auto stride_a = allocate_stride_tensor<StrideA>(groups, meta_options);
  auto stride_b = allocate_stride_tensor<StrideB>(groups, meta_options);
  auto stride_c = allocate_stride_tensor<StrideC>(groups, meta_options);
  auto stride_d = allocate_stride_tensor<StrideD>(groups, meta_options);

  build_hopper_metadata<Gemm, Element>(
      a,
      b,
      c,
      batch_sizes,
      trans_a,
      trans_b,
      problem_sizes,
      ptr_a,
      ptr_b,
      ptr_c,
      ptr_d,
      stride_a,
      stride_b,
      stride_c,
      stride_d,
      stream);

  run_hopper_grouped_gemm<Gemm>(
      problem_sizes,
      ptr_a,
      ptr_b,
      ptr_c,
      ptr_d,
      stride_a,
      stride_b,
      stride_c,
      stride_d,
      static_cast<int32_t>(groups),
      device_index,
      stream);
}

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC>
void run_hopper_variant(
    const std::string& variant,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b,
    int device_index,
    cudaStream_t stream) {
  if (variant == "wide_coop") {
    using Gemm = HopperGroupedGemmWide<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "wide_pingpong") {
    using Gemm = HopperGroupedGemmWidePingpong<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "tall_coop") {
    using Gemm = HopperGroupedGemmTall<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "tall_pingpong") {
    using Gemm = HopperGroupedGemmTallPingpong<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "square_coop") {
    using Gemm = HopperGroupedGemmSquareCooperative<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "square_pingpong") {
    using Gemm = HopperGroupedGemmSquarePingpong<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  if (variant == "smallm_pingpong") {
    using Gemm = HopperGroupedGemmSmallM<Element, LayoutA, LayoutB, LayoutC>;
    run_hopper_backend<Gemm, Element>(
        a, b, c, batch_sizes, trans_a, trans_b, device_index, stream);
    return;
  }
  TORCH_CHECK(
      false,
      "Invalid SM90 MoE grouped GEMM variant '",
      variant,
      "'. Expected one of: wide_coop, wide_pingpong, tall_coop, "
      "tall_pingpong, square_coop, square_pingpong, "
      "smallm_pingpong.");
}

#endif

template <typename Element>
void run_sm80_backend(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& batch_sizes,
    bool trans_a,
    bool trans_b,
    cudaStream_t stream) {
  auto meta_options = torch::TensorOptions().device(a.device()).dtype(torch::kInt64);
  auto problem_options = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
  int64_t groups = batch_sizes.size(0);

  auto problem_sizes = torch::empty({groups, 3}, problem_options);
  auto ptr_a = torch::empty({groups}, meta_options);
  auto ptr_b = torch::empty({groups}, meta_options);
  auto ptr_c = torch::empty({groups}, meta_options);
  auto ptr_d = torch::empty({groups}, meta_options);
  auto lda = torch::empty({groups}, meta_options);
  auto ldb = torch::empty({groups}, meta_options);
  auto ldc = torch::empty({groups}, meta_options);
  auto ldd = torch::empty({groups}, meta_options);

  build_metadata<Element>(
      a,
      b,
      c,
      batch_sizes,
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

  auto stream = at::cuda::getCurrentCUDAStream(a_contig.device().index()).stream();
  int device_index = a_contig.device().index();
  BackendPreference preference = backend_preference_from_env();
  bool use_sm90 = preference == BackendPreference::kSm90 ||
      (preference == BackendPreference::kAuto && device_supports_sm90(device_index));

  if (use_sm90) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    if (trans_a) {
      std::string variant = env_or_default(
          "NANOPLM_MOE_GROUPED_GEMM_TRANS_A_VARIANT", "wide_coop");
      run_hopper_variant<
          Element,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor>(
          variant,
          a_contig,
          b_contig,
          c,
          batch_sizes_contig,
          trans_a,
          trans_b,
          device_index,
          stream);
    } else if (trans_b) {
      const char* default_variant = b_contig.size(1) <= 512 ? "wide_coop" : "tall_coop";
      std::string variant = env_or_default(
          "NANOPLM_MOE_GROUPED_GEMM_TRANS_B_VARIANT", default_variant);
      run_hopper_variant<
          Element,
          cutlass::layout::RowMajor,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>(
          variant,
          a_contig,
          b_contig,
          c,
          batch_sizes_contig,
          trans_a,
          trans_b,
          device_index,
          stream);
    } else {
      if (a_contig.size(1) <= 512) {
        std::string variant = env_or_default(
            "NANOPLM_MOE_GROUPED_GEMM_WO_VARIANT", "tall_coop");
        run_hopper_variant<
            Element,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor>(
            variant,
            a_contig,
            b_contig,
            c,
            batch_sizes_contig,
            trans_a,
            trans_b,
            device_index,
            stream);
      } else {
        std::string variant = env_or_default(
            "NANOPLM_MOE_GROUPED_GEMM_WI_VARIANT", "tall_coop");
        run_hopper_variant<
            Element,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor,
            cutlass::layout::RowMajor>(
            variant,
            a_contig,
            b_contig,
            c,
            batch_sizes_contig,
            trans_a,
            trans_b,
            device_index,
            stream);
      }
    }
#else
    TORCH_CHECK(
        false,
        "NANOPLM_MOE_GROUPED_GEMM_BACKEND requested SM90, but this extension was "
        "not compiled with CUTLASS SM90 TMA support. Rebuild with CUDA 13 and "
        "compute_90/sm_90 gencode flags.");
#endif
  } else {
    run_sm80_backend<Element>(
        a_contig, b_contig, c, batch_sizes_contig, trans_a, trans_b, stream);
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
