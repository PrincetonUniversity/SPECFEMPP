#pragma once

#include "constants.hpp"
#include "enumerations/dimension.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::parallel_config {

template <specfem::dimension::type DimensionTag, int ChunkSize,
          typename ExecutionSpace>
struct edge_chunk_config {
  constexpr static auto dimension = DimensionTag; ///< Dimension type
  using execution_space = ExecutionSpace;         ///< Execution space
  constexpr static int chunk_size = ChunkSize;    ///< Number of edges per chunk
};

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
} // namespace specfem::parallel_config
