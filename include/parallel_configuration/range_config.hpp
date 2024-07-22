#pragma once

namespace specfem {
namespace parallel_config {

template <typename SIMDType> struct range_config { using simd = SIMDType; };

#ifdef KOKKOS_ENABLE_CUDA
template <typename SIMDType>
using default_range_config = range_config<SIMDType>;
#else
template <typename SIMDType>
using default_range_config = range_config<SIMDType>;
#endif

} // namespace parallel_config
} // namespace specfem
