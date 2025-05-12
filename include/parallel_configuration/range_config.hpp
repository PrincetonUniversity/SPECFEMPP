#pragma once

#include <cstddef>

namespace specfem {
namespace parallel_config {

/**
 * @brief Parallel configuration for range policy
 *
 * @tparam SIMD SIMD type @ref specfem::datatype::simd
 * @tparam ExecutionSpace Execution space
 */
template <typename SIMD, typename ExecutionSpace, std::size_t ChunkSize,
          std::size_t TileSize>
struct range_config {
  using simd = SIMD;
  using execution_space = ExecutionSpace;
  static constexpr bool is_point_parallel_config = true;
  constexpr static std::size_t chunk_size = ChunkSize;
  constexpr static std::size_t tile_size = TileSize;
};

/**
 * @brief Type alias for default range configuration
 *
 * @tparam SIMD SIMD type @ref specfem::datatype::simd
 * @tparam ExecutionSpace Execution space
 */
template <typename SIMD, typename ExecutionSpace>
using default_range_config = range_config<SIMD, ExecutionSpace, 1, 1>;

} // namespace parallel_config
} // namespace specfem
