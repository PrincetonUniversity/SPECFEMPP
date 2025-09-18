#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::chunk_element::impl {

// clang-format off
/**
 * @brief Chunk element field accessor for storing field values at all
 *        quadrature points within a chunk.
 *
 * This class provides an accessor interface for chunk-based element field data,
 * supporting efficient storage and access to field values at all
 * Gauss-Lobatto-Legendre (GLL) quadrature points within a chunk of elements.
 * It is designed for use in chunk execution policies where spacial locality of
 * data is critical for performance. An example usage is shown below:
 *
 * @code{.cpp}
 * // Compute the gradient of a field
 * using ChunkField = ...
 * int scratch_size = ChunkField::shmem_size();
 * specfem::execution::ChunkedDomainIterator chunk(...)
 * specfem::execution::for_each_level("compute_gradient",
 *     chunk.set_scratch_size(scratch_size),
 *     KOKKOS_LAMBDA(const decltype(chunk)::index_type &index) {
 *         ChunkField field(chunk.team_scratch(0));
 *         specfem::assembly::load_on_device(index.get_index(), field_container, field);
 *         specfem::algorithms::gradient(..., field, [&](..., gradient_value) {
 *             // Store the computed gradient value back to the field
 *         });
 *     });
 * @endcode
 *
 * @tparam ChunkSize     Number of elements processed together in a chunk.
 * @tparam NGLL          Number of Gauss-Lobatto-Legendre points per spatial dimension.
 * @tparam DimensionTag  Spatial dimension (dim2 or dim3) of the field.
 * @tparam MediumTag     Medium type (acoustic, elastic, poroelastic, etc.).
 * @tparam DataClass     Data class type for access control and memory traits.
 * @tparam UseSIMD       Whether to enable SIMD vectorization for performance.
 *
 */
// clang-format on
template <int ChunkSize, int NGLL, specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::data_access::DataClassType DataClass, bool UseSIMD>
class field : public specfem::data_access::Accessor<
                  specfem::data_access::AccessorType::chunk_element, DataClass,
                  DimensionTag, UseSIMD> {
private:
  /// @brief Type alias for the base accessor class
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::chunk_element, DataClass,
      DimensionTag, UseSIMD>;

public:
  /// @brief Number of field components based on dimension and medium type
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /// @brief Number of elements in the chunk
  constexpr static int nelements = ChunkSize;

  /// @brief Number of Gauss-Lobatto-Legendre points per spatial dimension
  constexpr static int ngll = NGLL;

  /// @brief SIMD type for vectorized operations
  using simd = typename base_type::template simd<type_real>;
  constexpr static auto using_simd = UseSIMD;

  /// @brief Vector type for storing chunk field data with optimized layout
  using value_type =
      typename base_type::template vector_type<type_real, nelements, ngll,
                                               components>;

  /// @brief Dimension tag identifying the physical medium type
  constexpr static auto dimension_tag = DimensionTag;

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
   * indexing. For chunk element fields, the indexing typically follows the
   * pattern: (element_id, gll_indices..., component_id).
   *
   * @tparam Indices Parameter pack for multi-dimensional indices
   * @param indices The indices specifying the location and component to access
   * @return Reference to the field value at the specified location
   *
   * @code{.cpp}
   * // For 2D fields: (element, i_gll, j_gll, component)
   * auto disp_x = field(ielem, i, j, 0);  // x-component
   * auto disp_z = field(ielem, i, j, 1);  // z-component
   * field(ielem, i, j, 0) = new_value;
   *
   * // For 3D fields: (element, i_gll, j_gll, k_gll, component)
   * auto value = field(ielem, i, j, k, icomp);
   * @endcode
   */
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(Indices... indices) const {
    return m_data(indices...);
  }

  KOKKOS_FORCEINLINE_FUNCTION auto &
  operator()(const specfem::point::index<dimension_tag, using_simd> &index) {
    return m_data(index);
  }

  KOKKOS_FORCEINLINE_FUNCTION typename value_type::value_type &
  operator()(const specfem::point::index<dimension_tag, using_simd> &index,
             const int &icomp) {
    return m_data(index, icomp);
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
   * @brief Access internal field data storage.
   *
   * @return const reference to the internal value_type storing field components
   */
  KOKKOS_INLINE_FUNCTION const value_type &get_data() const { return m_data; }
};

} // namespace specfem::chunk_element::impl
