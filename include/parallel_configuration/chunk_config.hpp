#pragma once

#include "constants.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace parallel_config {
/**
 * @brief Parallel configuration for chunk policy
 *
 * This policy is to be used along with element_chunk policy. It specifies the
 * number of threads (ChunkSize) and the number of vector lanes.
 *
 * @tparam NumThreads Number of threads per team
 * @tparam VectorLanes Number of vector lanes per thread
 */
template <int ChunkSize, int TileSize, int NumThreads, int VectorLanes,
          typename SIMDtype>
struct chunk_config {
  constexpr static int num_threads = NumThreads;   ///< Number of threads
  constexpr static int vector_lanes = VectorLanes; ///< Number of vector lanes
  constexpr static int tile_size = TileSize;       ///< Tile size
  constexpr static int chunk_size = ChunkSize;
  using simd = SIMDtype; ///< SIMD type
};

#ifdef KOKKOS_ENABLE_CUDA
constexpr static int chunk_size =
    specfem::build_configuration::chunk::chunk_size;
constexpr static int tile_size =
    specfem::build_configuration::chunk::chunk_size *
    specfem::build_configuration::chunk::num_chunks;
constexpr static int num_threads =
    specfem::build_configuration::chunk::num_threads;
constexpr static int vector_lanes =
    specfem::build_configuration::chunk::vector_lanes;
template <typename SIMDtype>
using default_chunk_config =
    chunk_config<chunk_size, tile_size, num_threads, vector_lanes, SIMDtype>;
#else
template <typename SIMDtype>
using default_chunk_config = chunk_config<1, 20, 1, 1, SIMDtype>;
#endif
} // namespace parallel_config
} // namespace specfem
