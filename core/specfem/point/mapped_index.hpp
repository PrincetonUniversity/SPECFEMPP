#pragma once

#include "enumerations/interface.hpp"
#include "index.hpp"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionTag, bool UseSIMD>
struct mapped_index : public index<DimensionTag, UseSIMD> {
private:
  using base_type = index<DimensionTag, UseSIMD>;

public:
  int imap; ///< Index of the mapped element

  KOKKOS_INLINE_FUNCTION
  mapped_index(const base_type &index, const int &imap)
      : base_type(index), imap(imap) {}
};

} // namespace point
} // namespace specfem
