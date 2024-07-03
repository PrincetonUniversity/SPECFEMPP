#pragma once

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
template <int ChunkSize, int TileSize, int NumThreads, int VectorLanes>
struct chunk_config {
  constexpr static int num_threads = NumThreads;   ///< Number of threads
  constexpr static int vector_lanes = VectorLanes; ///< Number of vector lanes
  constexpr static int tile_size = TileSize;       ///< Tile size
  constexpr static int chunk_size = ChunkSize;
};

#ifdef KOKKOS_ENABLE_CUDA
using default_chunk_config =
    chunk_config<32, 32 * 20, 160, 1>; ///< Default chunk configuration for CUDA
#else
using default_chunk_config =
    chunk_config<1, 20, 1, 1>; ///< Default chunk configuration for CPU
#endif
} // namespace parallel_config
} // namespace specfem
