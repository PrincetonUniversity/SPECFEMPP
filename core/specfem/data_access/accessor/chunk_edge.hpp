#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

/**
 * @brief Chunk-based edge accessor for vectorized interface operations.
 *
 * Provides SIMD-optimized data access for edge/interface computations.
 * Uses scratch memory for efficient chunked processing of edge data
 * with configurable vectorization.
 *
 * @tparam DataClass Type of edge data (intersection factors, normals, etc.)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::chunk_edge, DataClass,
                DimensionTag, UseSIMD> {
  /// @brief Accessor pattern identifier
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  /// @brief Data classification type
  constexpr static auto data_class = DataClass;
  /// @brief Spatial dimension
  constexpr static auto dimension_tag = DimensionTag;
  /// @brief SIMD vectorization flag
  constexpr static bool using_simd = UseSIMD;

  /**
   * @brief Datatype used to store data with optional SIMD vectorization
   *
   * @tparam T Base data type
   */
  template <typename T> using simd = specfem::datatype::simd<T, UseSIMD>;

  /**
   * @brief Scalar field storage for chunked edge elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element
   */
  template <typename T, int nelements, int ngll>
  using scalar_type =
      Kokkos::View<typename simd<T>::datatype[nelements][ngll],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  /**
   * @brief Vector field storage for chunked edge elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element
   * @tparam components Number of vector components
   */
  template <typename T, int nelements, int ngll, int components>
  using vector_type =
      Kokkos::View<typename simd<T>::datatype[nelements][ngll][components],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  /**
   * @brief Tensor field storage for chunked edge elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element
   * @tparam components Number of tensor components
   * @tparam dimension Spatial dimension
   */
  template <typename T, int nelements, int ngll, int components, int dimension>
  using tensor_type = Kokkos::View<
      typename simd<T>::datatype[nelements][ngll][components][dimension],
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
};

/**
 * @brief Type trait to detect chunk edge accessor types.
 */
template <typename T, typename = void>
struct is_chunk_edge : std::false_type {};

template <typename T>
struct is_chunk_edge<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::data_access::AccessorType::chunk_edge> >
    : std::true_type {};

} // namespace specfem::data_access
