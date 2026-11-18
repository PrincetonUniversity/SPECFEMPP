#pragma once

#include "specfem/experimental/layout/layout.hpp"

namespace specfem::testing::layout_transform_impl {
using restricted_base = decltype(specfem::experimental::layout::make_layout(
    specfem::experimental::layout::make_tuple(
        specfem::experimental::layout::make_tuple(2, 3), 2),
    specfem::experimental::layout::make_tuple(
        specfem::experimental::layout::make_tuple(1, 2), 6)));

// Three holes inside a rectangular envelope make inverse coordinate transport
// observable without adding a predicate constructor to the production API.
// Offsets 1, 10 and 11, original coordinates ((a, b), c) = ((1, 0), 0),
// ((0, 2), 1) and ((1, 2), 1), are invalid.
struct RestrictedMapping : restricted_base {
  KOKKOS_INLINE_FUNCTION RestrictedMapping();
  template <typename Coordinate>
  KOKKOS_INLINE_FUNCTION bool valid(const Coordinate &coordinate) const;
};
} // namespace specfem::testing::layout_transform_impl

KOKKOS_INLINE_FUNCTION
specfem::testing::layout_transform_impl::RestrictedMapping::RestrictedMapping()
    : restricted_base(specfem::experimental::layout::make_tuple(
                          specfem::experimental::layout::make_tuple(2, 3), 2),
                      specfem::experimental::layout::make_tuple(
                          specfem::experimental::layout::make_tuple(1, 2), 6)) {
}

template <typename Coordinate>
KOKKOS_INLINE_FUNCTION bool
specfem::testing::layout_transform_impl::RestrictedMapping::valid(
    const Coordinate &coordinate) const {
  if (!restricted_base::valid(coordinate)) {
    return false;
  }
  const auto offset = restricted_base::offset(coordinate);
  return offset != 1 && offset != 10 && offset != 11;
}
