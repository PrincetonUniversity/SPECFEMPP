#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

/**
 * @brief Chunk-based element accessor for vectorized domain operations.
 *
 * Provides SIMD-optimized data access for element-wise computations.
 * Uses specialized view types for efficient chunked processing of
 * element data with configurable vectorization.
 *
 * @tparam DataClass Type of element data (properties, fields, etc.)
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::chunk_element, DataClass,
                DimensionTag, UseSIMD> {
  /// @brief Accessor pattern identifier
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::chunk_element;
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
   * @brief Scalar field storage for chunked elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element dimension
   */
  template <typename T, int nelements, int ngll>
  using scalar_type =
      specfem::datatype::ScalarChunkElementViewType<T, DimensionTag, nelements,
                                                    ngll, UseSIMD>;

  /**
   * @brief Vector field storage for chunked elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element dimension
   * @tparam components Number of vector components
   */
  template <typename T, int nelements, int ngll, int components>
  using vector_type =
      specfem::datatype::VectorChunkElementViewType<T, DimensionTag, nelements,
                                                    ngll, components, UseSIMD>;

  /**
   * @brief Tensor field storage for chunked elements
   *
   * @tparam T Base data type
   * @tparam nelements Number of elements in the chunk
   * @tparam ngll Number of GLL points per element dimension
   * @tparam components Number of tensor components
   * @tparam dimension Spatial dimension
   */
  template <typename T, int nelements, int ngll, int components, int dimension>
  using tensor_type = specfem::datatype::TensorChunkElementViewType<
      T, DimensionTag, nelements, ngll, components, dimension, UseSIMD>;
};

/**
 * @brief Type trait to detect chunk element accessor types.
 */
template <typename T, typename = void>
struct is_chunk_element : std::false_type {};

template <typename T>
struct is_chunk_element<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::data_access::AccessorType::chunk_element> >
    : std::true_type {};

} // namespace specfem::data_access
