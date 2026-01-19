#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/execution.hpp"

namespace specfem {
namespace chunk_element {

/**
 * @brief Mapped index for chunked element operations with indirection.
 *
 * Provides index mapping capabilities for chunks of spectral elements,
 * combining execution context with data access patterns. Enables indirect
 * element access through mapping arrays for non-contiguous element processing.
 *
 * @tparam DimensionTag Spatial dimension (2D/3D)
 * @tparam SIMD SIMD configuration for vectorization
 * @tparam ViewType Kokkos view type for index storage
 * @tparam TeamMemberType Kokkos team execution context
 */
template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class MappedIndex : public specfem::execution::MappedChunkElementIndex<
                        DimensionTag, SIMD, ViewType, TeamMemberType>,
                    public specfem::data_access::Accessor<
                        specfem::data_access::AccessorType::chunk_element,
                        specfem::data_access::DataClassType::mapped_index,
                        DimensionTag, SIMD::using_simd> {
private:
  /// @brief Base execution index type for chunk element mapping
  using base_type =
      specfem::execution::MappedChunkElementIndex<DimensionTag, SIMD, ViewType,
                                                  TeamMemberType>;

public:
  /// @brief Iterator type from base class
  using iterator_type = typename base_type::iterator_type;

  /**
   * @brief Construct from base index.
   *
   * @param base Base mapped chunk element index
   */
  KOKKOS_INLINE_FUNCTION
  MappedIndex(const base_type &base) : base_type(base) {}

  /**
   * @brief Construct with explicit parameters.
   *
   * @param indices Element indices view
   * @param mapping Index mapping view for indirection
   * @param ngllz Number of GLL points in z-dimension
   * @param ngllx Number of GLL points in x-dimension
   * @param kokkos_index Team member execution context
   */
  KOKKOS_INLINE_FUNCTION
  MappedIndex(const ViewType indices, const ViewType mapping, const int &ngllz,
              const int &ngllx, const TeamMemberType &kokkos_index)
      : base_type(indices, mapping, ngllz, ngllx, kokkos_index) {}
};

} // namespace chunk_element
} // namespace specfem
