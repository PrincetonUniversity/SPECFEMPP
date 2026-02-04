#pragma once

#include "constants.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::parallel_configuration {

/**
 * @brief Parallel configuration for face chunk processing in 3D.
 *
 * @tparam DimensionTag Spatial dimension (dim3)
 * @tparam ChunkSize Number of faces processed per chunk
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = face_chunk_config<dim3, 32, Kokkos::Cuda>;
 * // Use config in specfem::execution::ChunkedFaceIterator<config>
 * @endcode
 *
 * @see specfem::execution
 * @see specfem::parallel_configuration::default_chunk_face_config
 */
template <specfem::dimension::type DimensionTag, int ChunkSize,
          typename ExecutionSpace>
struct face_chunk_config {
  constexpr static auto dimension = DimensionTag; ///< Dimension type
  using execution_space = ExecutionSpace;         ///< Execution space
  constexpr static int chunk_size = ChunkSize;    ///< Number of faces per chunk
};

/**
 * @brief Platform-optimized face chunk configuration defaults.
 *
 * Automatically selects optimal face chunk sizes based on execution space:
 * - CUDA: 16 faces per chunk (smaller than edges due to 2D surface)
 * - HIP: 32 faces per chunk
 * - OpenMP/Serial: 1 face per chunk
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = default_chunk_face_config<dim3, Kokkos::Cuda>;
 * // Automatically uses chunk_size=16 for CUDA
 * @endcode
 *
 * @see specfem::execution
 */
template <specfem::dimension::type DimensionTag, typename ExecutionSpace>
struct default_chunk_face_config;

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct default_chunk_face_config<specfem::dimension::type::dim3, Kokkos::Cuda>
    : face_chunk_config<specfem::dimension::type::dim3, 4, Kokkos::Cuda> {};
#endif

#if defined(KOKKOS_ENABLE_HIP)

template <>
struct default_chunk_face_config<specfem::dimension::type::dim3, Kokkos::HIP>
    : face_chunk_config<specfem::dimension::type::dim3, 8, Kokkos::HIP> {};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
struct default_chunk_face_config<specfem::dimension::type::dim3, Kokkos::OpenMP>
    : face_chunk_config<specfem::dimension::type::dim3, 1, Kokkos::OpenMP> {};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
template <>
struct default_chunk_face_config<specfem::dimension::type::dim3, Kokkos::Serial>
    : face_chunk_config<specfem::dimension::type::dim3, 1, Kokkos::Serial> {};

template <>
struct default_chunk_face_config<specfem::dimension::type::dim3,
                                 Kokkos::HostSpace>
    : default_chunk_face_config<specfem::dimension::type::dim3,
                                Kokkos::Serial> {};

#endif
} // namespace specfem::parallel_configuration
