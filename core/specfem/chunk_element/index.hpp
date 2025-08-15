#pragma once

#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "specfem/data_access.hpp"

namespace specfem {
namespace chunk_element {

template <specfem::dimension::type DimensionTag, typename SIMD,
          typename ViewType, typename TeamMemberType>
class Index
    : public specfem::execution::ChunkElementIndex<DimensionTag, SIMD, ViewType,
                                                   TeamMemberType>,
      public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_element,
          specfem::data_access::DataClassType::index, DimensionTag,
          SIMD::using_simd> {
private:
  using base_type =
      specfem::execution::ChunkElementIndex<DimensionTag, SIMD, ViewType,
                                            TeamMemberType>; ///< Base type of
                                                             ///< the chunk
                                                             ///< element index
public:
  using iterator_type = typename base_type::iterator_type;

  KOKKOS_INLINE_FUNCTION
  Index(const base_type &base) : base_type(base) {}

  KOKKOS_INLINE_FUNCTION
  Index(const ViewType indices, const int &ngllz, const int &ngllx,
        const TeamMemberType &kokkos_index)
      : base_type(indices, ngllz, ngllx, kokkos_index) {}
};

} // namespace chunk_element
} // namespace specfem
