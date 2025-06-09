#pragma once

#include "enumerations/interface.hpp"
#include "index.hpp"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct mapped_index : public index<DimensionTag, UseSIMD> {
private:
  using base_type = index<DimensionTag, UseSIMD>;
  using accessor_type =
      specfem::accessor::Accessor<specfem::accessor::type::point,
                                  specfem::data_class::type::mapped_index,
                                  DimensionTag, UseSIMD>; ///< Accessor type for
                                                          ///< mapped index

public:
  int imap; ///< Index of the mapped element

  constexpr static auto data_class =
      accessor_type::data_class; ///< Data class of the mapped index

  KOKKOS_INLINE_FUNCTION
  mapped_index(const base_type &index, const int &imap)
      : base_type(index), imap(imap) {}
};

} // namespace point
} // namespace specfem
