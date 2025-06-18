#pragma once

#include "enumerations/interface.hpp"
#include "execution/mapped_chunked_domain_iterator.hpp"

namespace specfem {
namespace chunk_element {

template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class MappedIndex
    : public specfem::execution::MappedChunkElementIndex<
          DimensionTag, SIMD, ViewType, TeamMemberType>,
      public specfem::accessor::Accessor<
          specfem::accessor::type::chunk_element,
          specfem::data_class::type::mapped_index, DimensionTag, SIMD::value> {
private:
  using base_type =
      specfem::execution::MappedChunkElementIndex<DimensionTag, SIMD, ViewType,
                                                  TeamMemberType>; ///< Base
                                                                   ///< type of
                                                                   ///< the
                                                                   ///< chunk
                                                                   ///< element
                                                                   ///< index
  using accessor_type =
      specfem::accessor::Accessor<specfem::accessor::type::chunk_element,
                                  specfem::data_class::type::mapped_index,
                                  DimensionTag, SIMD::value>; ///< Accessor type
public:
  using iterator_type = typename base_type::iterator_type;

  KOKKOS_INLINE_FUNCTION
  MappedIndex(const base_type &base) : base_type(base) {}

  KOKKOS_INLINE_FUNCTION
  MappedIndex(const ViewType indices, const ViewType mapping, const int &ngllz,
              const int &ngllx, const TeamMemberType &kokkos_index)
      : base_type(indices, mapping, ngllz, ngllx, kokkos_index) {}
};

} // namespace chunk_element
} // namespace specfem
