#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::chunk_edge::impl {

/**
 * @brief Accessor-agnostic field implementation for chunk-based edge data.
 *
 * This class provides a accessor-agnostic implementation for chunk-based
 * edge data, primarily used when making unified references to different
 * field types. See the code example below :
 *
 * @code{.cpp}
 * // Example usage of field_without_accessor
 * const specfem::wavefield::type wavefield_type = ...;
 * const auto field = [&]() {
 *   if (wavefield_type == specfem::wavefield::type::displacement) {
 *     return displacement_field.field_without_accessor();
 *   } else if (wavefield_type == specfem::wavefield::type::velocity) {
 *     return velocity_field.field_without_accessor();
 *   } else if (wavefield_type == specfem::wavefield::type::acceleration) {
 *     return acceleration_field.field_without_accessor();
 *   } else {
 *     throw std::runtime_error("Unsupported wavefield type");
 *   }
 * }();
 * @endcode
 *
 * @note Note that this class is used internally and external references to it
 *       should be made using @c auto keyword.
 *
 * @tparam ChunkSize Number of edges processed together in a chunk
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points per spatial dimension
 * @tparam DimensionTag Spatial dimension (dim2 or dim3) of the field
 * @tparam MediumTag Medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization for performance
 * @tparam ValueType The underlying data storage type for field values
 */
template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD,
          typename ValueType>
class field_without_accessor {
public:
  /// @brief Number of field components based on dimension and medium type
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /// @brief Number of edges in the chunk
  constexpr static int nedges = ChunkSize;

  /// @brief Number of Gauss-Lobatto-Legendre points per spatial dimension
  constexpr static int ngll = NGLL;

  /// @brief Medium tag identifying the physical medium type
  constexpr static auto medium_tag = MediumTag;

  /// @brief Type alias for the underlying data storage
  using value_type = ValueType;

  /// @brief SIMD type for vectorized operations
  using simd = specfem::datatype::simd<type_real, UseSIMD>;

private:
  /// @brief Internal storage for chunk field data
  value_type m_data;

public:
  /**
   * @brief Default constructor - creates field with uninitialized data.
   */
  KOKKOS_FORCEINLINE_FUNCTION field_without_accessor() = default;

  /**
   * @brief Construct field from existing data storage.
   *
   * Initializes the chunk field accessor with pre-allocated data storage.
   * This is typically used when wrapping existing Kokkos views or other
   * data structures.
   *
   * @param data_in The data storage object to wrap
   */
  KOKKOS_FORCEINLINE_FUNCTION
  field_without_accessor(const value_type &data_in) : m_data(data_in) {}

  /**
   * @brief Multi-dimensional index operator for accessing field components.
   *
   * Provides access to individual field values using multi-dimensional
   * indexing. The exact indexing scheme depends on the ValueType
   * implementation, but typically follows the pattern: (edge_id,
   * gll_indices..., component_id).
   *
   * @tparam Indices Parameter pack for multi-dimensional indices
   * @param indices The indices specifying the location and component to access
   * on the edge
   * @return Reference to the field value at the specified location
   */
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(Indices... indices) const {
    return m_data(indices...);
  }
};

/**
 * @brief Type trait helper for removing accessor attributes from field types.
 *
 * This template metafunction provides a type alias that maps from the full
 * field type (with accessor interface) to the lightweight
 * field_without_accessor type. It is used for type transformations when the
 * accessor interface overhead is not needed.
 *
 * @tparam ChunkSize Number of edges processed together in a chunk
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points per spatial dimension
 * @tparam DimensionTag Spatial dimension (dim2 or dim3) of the field
 * @tparam MediumTag Medium type (acoustic, elastic, poroelastic, etc.)
 * @tparam UseSIMD Whether to enable SIMD vectorization
 * @tparam ValueType The underlying data storage type
 *
 * @see field_without_accessor for the resulting type
 *
 */
template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD,
          typename ValueType>
class remove_accessor_attribute {
public:
  /// @brief Type alias for the field without accessor interface
  using type = field_without_accessor<ChunkSize, NGLL, DimensionTag, MediumTag,
                                      UseSIMD, ValueType>;
};

/**
 * @brief Chunk edge field accessor for storing field values at all
 *        quadrature points within a chunk.
 *
 * This class provides an accessor interface for chunk-based edge field data,
 * supporting efficient storage and access to field values at all
 * Gauss-Lobatto-Legendre (GLL) quadrature points within a chunk of edges.
 * It is designed for use in chunk execution policies where spacial locality of
 * data is critical for performance. An example usage is shown below:
 *
 * @tparam ChunkSize     Number of edges processed together in a chunk.
 * @tparam NGLL          Number of Gauss-Lobatto-Legendre points per spatial
 * dimension.
 * @tparam DimensionTag  Spatial dimension (dim2 or dim3) of the field.
 * @tparam MediumTag     Medium type (acoustic, elastic, poroelastic, etc.).
 * @tparam DataClass     Data class type for access control and memory traits.
 * @tparam UseSIMD       Whether to enable SIMD vectorization for performance.
 *
 * @see remove_accessor_attribute for type trait to strip accessor attributes.
 */
template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::chunk_edge, DataClass,
                  DimensionTag, UseSIMD> {
private:
  /// @brief Type alias for the base accessor class
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_edge, DataClass, DimensionTag,
      UseSIMD>;

public:
  /// @brief Number of field components based on dimension and medium type
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /// @brief Number of edges in the chunk
  constexpr static int nedges = ChunkSize;

  /// @brief Number of Gauss-Lobatto-Legendre points per spatial dimension
  constexpr static int ngll = NGLL;

  /// @brief SIMD type for vectorized operations
  using simd = typename base_type::template simd<type_real>;

  /// @brief Vector type for storing chunk field data with optimized layout
  using value_type = typename base_type::template vector_type<type_real, nedges,
                                                              ngll, components>;

  /// @brief Medium tag identifying the physical medium type
  constexpr static auto medium_tag = MediumTag;

private:
  /// @brief Internal storage for chunk field data
  value_type m_data;

public:
  /**
   * @brief Default constructor - creates field with uninitialized data.
   */
  KOKKOS_FORCEINLINE_FUNCTION field() = default;

  /**
   * @brief Construct field with scratch memory allocation.
   *
   * Initializes the chunk field with memory allocated from the provided scratch
   * memory space. This is commonly used in team-based Kokkos execution where
   * scratch memory is shared among team members for efficient data access.
   *
   * @tparam ScratchMemorySpace Type of the scratch memory space (Kokkos
   * concept)
   * @param scratch_space The scratch memory space to allocate from
   *
   * @code{.cpp}
   * // Example usage in a Kokkos team lambda
   * KOKKOS_LAMBDA(const TeamType& team) {
   *   DisplacementField field(team.team_scratch(0));
   *   // Use field for computations...
   * }
   * @endcode
   */
  template <typename ScratchMemorySpace>
  KOKKOS_FORCEINLINE_FUNCTION field(const ScratchMemorySpace &scratch_space)
      : m_data(scratch_space) {}

  /**
   * @brief Multi-dimensional index operator for accessing field components.
   *
   * Provides access to individual field values using multi-dimensional
   * indexing. For chunk edge fields, the indexing typically follows the
   * pattern: (edge_id, gll_indices..., component_id).
   *
   * @tparam Indices Parameter pack for multi-dimensional indices
   * @param indices The indices specifying the location and component to access
   * @return Reference to the field value at the specified location
   *
   * @code{.cpp}
   * // For 2D fields: (edge, point_on_edge, component)
   * auto disp_x = field(ielem, ipoint, 0);  // x-component
   * auto disp_z = field(ielem, ipoint, 1);  // z-component
   * field(ielem, ipoint, 0) = new_value;
   * @endcode
   */
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(Indices... indices) const {
    return m_data(indices...);
  }

  /**
   * @brief Get the shared memory size required for this field type.
   *
   * Returns the amount of shared memory (in bytes) required to allocate
   * storage for this field type. This is used for Kokkos team scratch
   * memory allocation.
   *
   * @return Size in bytes required for field storage
   *
   * @code{.cpp}
   * // Example: Allocating team scratch memory
   * auto policy = Kokkos::TeamPolicy<>(num_teams, team_size)
   *   .set_scratch_size(0, Kokkos::PerTeam(DisplacementField::shmem_size()));
   * @endcode
   */
  constexpr static std::size_t shmem_size() { return value_type::shmem_size(); }

  /**
   * @brief Strip accessor attributes and return a @c field_without_accessor
   * type.
   *
   * This method provides a way to a unified field type that does not
   * include the accessor interface. It is useful when you need to reference
   * field of different types.
   *
   * @code{.cpp}
   * // Example usage of field_without_accessor
   * const specfem::wavefield::type wavefield_type = ...;
   * const auto field = [&]() {
   *   if (wavefield_type == specfem::wavefield::type::displacement) {
   *     return displacement_field.field_without_accessor();
   *   } else if (wavefield_type == specfem::wavefield::type::velocity) {
   *     return velocity_field.field_without_accessor();
   *   } else if (wavefield_type == specfem::wavefield::type::acceleration) {
   *     return acceleration_field.field_without_accessor();
   *   } else {
   *     throw std::runtime_error("Unsupported wavefield type");
   *   }
   * }();
   * @endcode
   *
   * @return A field_without_accessor type with the same data storage
   */
  KOKKOS_INLINE_FUNCTION const typename remove_accessor_attribute<
      ChunkSize, NGLL, DimensionTag, MediumTag, UseSIMD, value_type>::type
  field_without_accessor() const {
    return { m_data };
  }
};

} // namespace specfem::chunk_edge::impl
