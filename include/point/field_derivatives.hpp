#ifndef _POINT_FIELD_DERIVATIVES_HPP
#define _POINT_FIELD_DERIVATIVES_HPP

#include "datatypes/point_view.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
struct field_derivatives {
  static constexpr int components =
      specfem::medium::medium<DimensionType, MediumTag>::components;

  static constexpr int dimensions =
      specfem::dimension::dimension<DimensionType>::dim;

  using ViewType =
      specfem::datatype::VectorPointViewType<type_real, dimensions, components>;

  ViewType du;

  KOKKOS_FUNCTION field_derivatives() = default;

  KOKKOS_FUNCTION field_derivatives(const ViewType &du) : du(du) {}
};

} // namespace point
} // namespace specfem

#endif /* _POINT_FIELD_DERIVATIVES_HPP */
