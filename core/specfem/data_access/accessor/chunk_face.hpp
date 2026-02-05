#pragma once

#include "specfem/element.hpp"

#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

/**
 * @brief Chunk-based face accessor for vectorized interface operations in 3D.
 *
 * Provides SIMD-optimized data access for face/interface computations in 3D.
 * Uses scratch memory for efficient chunked processing of face data
 * with configurable vectorization. Faces are 2D surfaces with ngll x ngll
 * quadrature points.
 *
 * @tparam DataClass Type of face data (intersection factors, normals, etc.)
 * @tparam DimensionTag Spatial dimension (3D)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::data_access::DataClassType DataClass,
          specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct Accessor<specfem::datatype::AccessorType::chunk_face, DataClass,
                DimensionTag, UseSIMD> {
  /// @brief Accessor pattern identifier
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::chunk_face;
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
   * @brief Scalar field storage for chunked face elements
   *
   * @tparam T Base data type
   * @tparam nfaces Number of faces in the chunk
   * @tparam ngll Number of GLL points per face dimension
   */
  template <typename T, int nfaces, int ngll>
  using scalar_type =
      Kokkos::View<typename simd<T>::datatype[nfaces][ngll][ngll],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  /**
   * @brief Vector field storage for chunked face elements
   *
   * @tparam T Base data type
   * @tparam nfaces Number of faces in the chunk
   * @tparam ngll Number of GLL points per face dimension
   * @tparam components Number of vector components
   */
  template <typename T, int nfaces, int ngll, int components>
  using vector_type =
      Kokkos::View<typename simd<T>::datatype[nfaces][ngll][ngll][components],
                   Kokkos::DefaultExecutionSpace::scratch_memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  /**
   * @brief Tensor field storage for chunked face elements
   *
   * @tparam T Base data type
   * @tparam nfaces Number of faces in the chunk
   * @tparam ngll Number of GLL points per face dimension
   * @tparam components Number of tensor components
   * @tparam dimension Spatial dimension
   */
  template <typename T, int nfaces, int ngll, int components, int dimension>
  using tensor_type = Kokkos::View<
      typename simd<T>::datatype[nfaces][ngll][ngll][components][dimension],
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
};

/**
 * @brief Type trait to detect chunk face accessor types.
 */
template <typename T, typename = void>
struct is_chunk_face : std::false_type {};

template <typename T>
struct is_chunk_face<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::datatype::AccessorType::chunk_face> >
    : std::true_type {};

} // namespace specfem::data_access
