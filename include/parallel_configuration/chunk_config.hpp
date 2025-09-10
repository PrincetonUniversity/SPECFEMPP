#pragma once

#include "constants.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace parallel_config {

namespace impl {
constexpr int cuda_chunk_size = 32;
constexpr int hip_chunk_size = 64;
constexpr int openmp_chunk_size = 1;
constexpr int serial_chunk_size = 1;
} // namespace impl

#if defined(KOKKOS_ENABLE_CUDA)
constexpr int storage_chunk_size = impl::cuda_chunk_size;
constexpr int chunk_size = impl::cuda_chunk_size;
#elif defined(KOKKOS_ENABLE_HIP)
constexpr int storage_chunk_size = impl::hip_chunk_size;
constexpr int chunk_size = impl::hip_chunk_size;
#elif defined(KOKKOS_ENABLE_OPENMP)
constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();
constexpr int storage_chunk_size = impl::openmp_chunk_size * simd_size;
constexpr int chunk_size = impl::openmp_chunk_size;
#else
constexpr int simd_size = specfem::datatype::simd<type_real, true>::size();
constexpr int storage_chunk_size = impl::serial_chunk_size * simd_size;
constexpr int chunk_size = impl::serial_chunk_size;
#endif

/**
 * @brief Parallel configuration for chunk policy.
 *
 * @tparam DimensionTag Dimension type of the elements within a chunk.
 * @tparam ChunkSize Number of elements within a chunk.
 * @tparam TileSize Tile size for chunk policy.
 * @tparam NumThreads Number of threads within a team.
 * @tparam VectorLanes Number of vector lanes.
 * @tparam SIMD SIMD type to use simd operations. @ref specfem::datatypes::simd
 */
template <specfem::dimension::type DimensionTag, int ChunkSize, int TileSize,
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
      DimensionTag; ///< Dimension type of the elements within chunk.
};

/**
 * @brief Default chunk configuration to use based on Dimension, SIMD type and
 * Execution space.
 *
 * Defines chunk size, tile size, number of threads, number of vector lanes
 * defaults for @ref specfem::parallel_config::chunk_config
 *
 * @tparam DimensionTag Dimension type of the elements within a chunk.
 * @tparam SIMD SIMD type to use simd operations. @ref specfem::datatypes::simd
 * @tparam ExecutionSpace Execution space for the policy.
 */
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ExecutionSpace>
struct default_chunk_config;

#if defined(KOKKOS_ENABLE_CUDA)
template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD, Kokkos::Cuda>
    : chunk_config<specfem::dimension::type::dim2, impl::cuda_chunk_size,
                   impl::cuda_chunk_size, 512, 1, SIMD, Kokkos::Cuda> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD, Kokkos::Cuda>
    : chunk_config<specfem::dimension::type::dim3, impl::cuda_chunk_size,
                   impl::cuda_chunk_size, 512, 1, SIMD, Kokkos::Cuda> {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD, Kokkos::HIP>
    : chunk_config<specfem::dimension::type::dim2, impl::cuda_chunk_size,
                   impl::hip_chunk_size, 512, 1, SIMD, Kokkos::HIP> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD, Kokkos::HIP>
    : chunk_config<specfem::dimension::type::dim3, impl::hip_chunk_size,
                   impl::hip_chunk_size, 512, 1, SIMD, Kokkos::HIP> {};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD,
                            Kokkos::OpenMP>
    : chunk_config<specfem::dimension::type::dim2, impl::openmp_chunk_size,
                   impl::openmp_chunk_size, 1, 1, SIMD, Kokkos::OpenMP> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD,
                            Kokkos::OpenMP>
    : chunk_config<specfem::dimension::type::dim3, impl::openmp_chunk_size,
                   impl::openmp_chunk_size, 1, 1, SIMD, Kokkos::OpenMP> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD,
                            Kokkos::HostSpace>
    : default_chunk_config<specfem::dimension::type::dim2, SIMD,
                           Kokkos::OpenMP> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD,
                            Kokkos::HostSpace>
    : default_chunk_config<specfem::dimension::type::dim3, SIMD,
                           Kokkos::OpenMP> {};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD,
                            Kokkos::Serial>
    : chunk_config<specfem::dimension::type::dim2, impl::serial_chunk_size,
                   impl::serial_chunk_size, 1, 1, SIMD, Kokkos::Serial> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD,
                            Kokkos::Serial>
    : chunk_config<specfem::dimension::type::dim3, impl::serial_chunk_size,
                   impl::serial_chunk_size, 1, 1, SIMD, Kokkos::Serial> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim2, SIMD,
                            Kokkos::HostSpace>
    : default_chunk_config<specfem::dimension::type::dim2, SIMD,
                           Kokkos::Serial> {};

template <typename SIMD>
struct default_chunk_config<specfem::dimension::type::dim3, SIMD,
                            Kokkos::HostSpace>
    : default_chunk_config<specfem::dimension::type::dim3, SIMD,
                           Kokkos::Serial> {};
#endif
} // namespace parallel_config
} // namespace specfem
