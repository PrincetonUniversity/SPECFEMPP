#pragma once

#include "constants.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::parallel_configuration {

/**
 * @brief Parallel configuration for edge chunk processing.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam ChunkSize Number of edges processed per chunk
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = edge_chunk_config<dim2, 32, Kokkos::Cuda>;
 * // Use config in specfem::execution::ChunkedEdgeIterator<config>
 * @endcode
 *
 * @see specfem::execution
 * @see specfem::parallel_configuration::default_chunk_edge_config
 */
template <specfem::dimension::type DimensionTag, int ChunkSize,
          typename ExecutionSpace>
struct edge_chunk_config {
  constexpr static auto dimension = DimensionTag; ///< Dimension type
  using execution_space = ExecutionSpace;         ///< Execution space
  constexpr static int chunk_size = ChunkSize;    ///< Number of edges per chunk
};

/**
 * @brief Platform-optimized edge chunk configuration defaults.
 *
 * Automatically selects optimal edge chunk sizes based on execution space:
 * - CUDA: 32 edges per chunk
 * - HIP: 64 edges per chunk
 * - OpenMP/Serial: 1 edge per chunk
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = default_chunk_edge_config<dim2, Kokkos::Cuda>;
 * // Automatically uses chunk_size=32 for CUDA
 * @endcode
 *
 * @see specfem::execution
 */
template <specfem::dimension::type DimensionTag, typename ExecutionSpace>
struct default_chunk_edge_config;

#if defined(KOKKOS_ENABLE_CUDA)
template <specfem::dimension::type DimensionTag>
struct default_chunk_edge_config<DimensionTag, Kokkos::Cuda>
    : edge_chunk_config<DimensionTag, 32, Kokkos::Cuda> {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <specfem::dimension::type DimensionTag>
struct default_chunk_edge_config<DimensionTag, Kokkos::HIP>
    : edge_chunk_config<DimensionTag, 64, Kokkos::HIP> {};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <specfem::dimension::type DimensionTag>
struct default_chunk_edge_config<DimensionTag, Kokkos::OpenMP>
    : edge_chunk_config<DimensionTag, 1, Kokkos::OpenMP> {};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
template <specfem::dimension::type DimensionTag>
struct default_chunk_edge_config<DimensionTag, Kokkos::Serial>
    : edge_chunk_config<DimensionTag, 1, Kokkos::Serial> {};

template <specfem::dimension::type DimensionTag>
struct default_chunk_edge_config<DimensionTag, Kokkos::HostSpace>
    : default_chunk_edge_config<DimensionTag, Kokkos::Serial> {};
#endif
} // namespace specfem::parallel_configuration
