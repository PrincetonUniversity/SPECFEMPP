#pragma once

namespace specfem {
namespace parallel_config {

/**
 * @brief Parallel configuration for edge policy.
 *
 * @tparam DimensionTag Dimension type of the elements where the edge is
 * defined.
 * @tparam NumThreads Number of threads to use.
 * @tparam VectorLanes Number of vector lanes to use.
 * @tparam ExecutionSpace Execution space to use.
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
 * @brief Default edge configuration.
 *
 * Sets the number of threads and vector lanes to use for the edge policy based
 * on DimensionTag and ExecutionSpace.
 *
 * @tparam DimensionTag Dimension type of the elements where the edge is
 * defined.
 * @tparam ExecutionSpace Execution space to use.
 */
template <specfem::dimension::type DimensionTag, typename ExecutionSpace>
struct default_edge_config;

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::Cuda>
    : edge_config<specfem::dimension::type::dim2, 32, 1, Kokkos::Cuda> {};
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::OpenMP>
    : edge_config<specfem::dimension::type::dim2, 1, 1, Kokkos::OpenMP> {};
#endif

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct default_edge_config<specfem::dimension::type::dim2, Kokkos::Serial>
    : edge_config<specfem::dimension::type::dim2, 1, 1, Kokkos::Serial> {};
#endif

} // namespace parallel_config
} // namespace specfem
