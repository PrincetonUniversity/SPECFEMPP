#pragma once

#include "constants.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace parallel_config {
/**
 * @brief Parallel configuration for chunk policy.
 *
 * @tparam DimensionType Dimension type of the elements within a chunk.
 * @tparam ChunkSize Number of elements within a chunk.
 * @tparam TileSize Tile size for chunk policy.
 * @tparam NumThreads Number of threads within a team.
 * @tparam VectorLanes Number of vector lanes.
 * @tparam SIMD SIMD type to use simd operations. @ref specfem::datatypes::simd
 */
template <specfem::dimension::type DimensionType, int ChunkSize, int TileSize,
          int NumThreads, int VectorLanes, typename SIMD,
          typename ExecutionSpace>
struct chunk_config {
  constexpr static int num_threads = NumThreads;   ///< Number of threads
  constexpr static int vector_lanes = VectorLanes; ///< Number of vector lanes
  constexpr static int tile_size = TileSize;       ///< Tile size
  constexpr static int chunk_size = ChunkSize;     ///< Chunk size
  using simd = SIMD;                               ///< SIMD type
  using execution_space = ExecutionSpace;          ///< Execution space
  constexpr static auto dimension =
      DimensionType; ///< Dimension type of the elements within chunk.
};

/**
 * @brief Default chunk configuration to use based on Dimension, SIMD type and
 * Execution space.
 *
 * Defines chunk size, tile size, number of threads, number of vector lanes
 * defaults for @ref specfem::parallel_config::chunk_config
 *
 * @tparam DimensionType Dimension type of the elements within a chunk.
 * @tparam SIMD SIMD type to use simd operations. @ref specfem::datatypes::simd
 * @tparam ExecutionSpace Execution space for the policy.
 */
template <specfem::dimension::type DimensionType, typename SIMD,
          typename ExecutionSpace>
struct default_chunk_config;

#ifdef KOKKOS_ENABLE_CUDA
template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD, Kokkos::Cuda>
    : chunk_config<specfem::dimension::type::dim2, 32, 32, 160, 1, SIMD,
                   Kokkos::Cuda> {};
#endif

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD,
                            Kokkos::Serial>
    : chunk_config<specfem::dimension::type::dim2, 1, 1, 1, 1, SIMD,
                   Kokkos::Serial> {};
} // namespace parallel_config
} // namespace specfem
