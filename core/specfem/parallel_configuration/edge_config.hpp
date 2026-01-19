#pragma once

namespace specfem {
namespace parallel_configuration {

/**
 * @brief Parallel execution configuration for edge operations.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam NumThreads Thread count per team
 * @tparam VectorLanes Vector lane width
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = edge_config<dim2, 32, 1, Kokkos::Cuda>;
 * // Use config in specfem::execution::EdgeIterator<config>
 * @endcode
 *
 * @see specfem::execution
 * @see specfem::parallel_configuration::default_edge_config
 */
template <specfem::dimension::type DimensionTag, int NumThreads,
          int VectorLanes, typename ExecutionSpace>
struct edge_config {
  constexpr static auto dimension = DimensionTag;  ///< Dimension type
  using execution_space = ExecutionSpace;          ///< Execution space
  constexpr static int num_threads = NumThreads;   ///< Number of threads
  constexpr static int vector_lanes = VectorLanes; ///< Vector lanes
};

/**
 * @brief Platform-optimized edge configuration defaults.
 *
 * Automatically selects optimal thread and vector lane counts:
 * - CUDA/HIP: 32 threads, 1 vector lane
 * - OpenMP/Serial: 1 thread, 1 vector lane
 *
 * @tparam DimensionTag Spatial dimension
 * @tparam ExecutionSpace Kokkos execution space
 *
 * @code
 * using config = default_edge_config<dim2, Kokkos::Cuda>;
 * // Automatically uses: num_threads=32, vector_lanes=1 for CUDA
 * @endcode
 */
template <specfem::dimension::type DimensionTag, typename ExecutionSpace>
struct default_edge_config;

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::Cuda>
    : edge_config<specfem::dimension::type::dim2, 32, 1, Kokkos::Cuda> {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::HIP>
    : edge_config<specfem::dimension::type::dim2, 32, 1, Kokkos::HIP> {};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::OpenMP>
    : edge_config<specfem::dimension::type::dim2, 1, 1, Kokkos::OpenMP> {};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::Serial>
    : edge_config<specfem::dimension::type::dim2, 1, 1, Kokkos::Serial> {};
#endif

} // namespace parallel_configuration
} // namespace specfem
