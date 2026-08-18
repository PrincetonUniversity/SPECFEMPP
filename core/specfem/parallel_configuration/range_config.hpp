#pragma once

#include <cstddef>

namespace specfem {
namespace parallel_configuration {

/**
 * @brief Configuration for parallel range-based operations.
 *
 * @tparam SIMD SIMD vectorization type
 * @tparam ExecutionSpace Kokkos execution space
 * @tparam ChunkSize Elements per processing chunk
 * @tparam TileSize Memory tiling size
 *
 * @code
 * using config = range_config<simd_type, Kokkos::Cuda, 1024, 32>;
 * specfem::execution::RangeIterator<config> iterator;
 * @endcode
 *
 * @see specfem::execution::RangeIterator
 * @see specfem::parallel_configuration::default_range_config
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
 * @brief Default range configuration with minimal chunking.
 *
 * Uses chunk_size=1 and tile_size=1 for simple point-wise operations.
 *
 * @tparam SIMD SIMD vectorization type
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = default_range_config<simd_type, Kokkos::OpenMP>;
 * // Automatically uses: chunk_size=1, tile_size=1
 * @endcode
 */
template <typename SIMD, typename ExecutionSpace>
using default_range_config = range_config<SIMD, ExecutionSpace, 1, 1>;

} // namespace parallel_configuration
} // namespace specfem
