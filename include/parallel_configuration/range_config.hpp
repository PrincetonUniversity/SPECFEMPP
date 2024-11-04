#pragma once

namespace specfem {
namespace parallel_config {

/**
 * @brief Parallel configuration for range policy
 *
 * @tparam SIMD SIMD type @ref specfem::datatype::simd
 * @tparam ExecutionSpace Execution space
 */
template <typename SIMD, typename ExecutionSpace> struct range_config {
  using simd = SIMD;
  using execution_space = ExecutionSpace;
  static constexpr bool is_point_parallel_config = true;
};

/**
 * @brief Type alias for default range configuration
 *
 * @tparam SIMD SIMD type @ref specfem::datatype::simd
 * @tparam ExecutionSpace Execution space
 */
template <typename SIMD, typename ExecutionSpace>
using default_range_config = range_config<SIMD, ExecutionSpace>;

} // namespace parallel_config
} // namespace specfem
