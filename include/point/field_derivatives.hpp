#ifndef _POINT_FIELD_DERIVATIVES_HPP
#define _POINT_FIELD_DERIVATIVES_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace point {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag>
struct field_derivatives;

template <specfem::element::medium_tag MediumTag>
struct field_derivatives<specfem::dimension::type::dim2, MediumTag> {

  constexpr static int components =
      specfem::medium::medium<specfem::dimension::type::dim2,
                              MediumTag>::components;
  specfem::kokkos::array_type<type_real, components> du_dx;
  specfem::kokkos::array_type<type_real, components> du_dz;

  field_derivatives() = default;
  field_derivatives(
      const specfem::kokkos::array_type<type_real, components> du_dx,
      const specfem::kokkos::array_type<type_real, components> du_dz)
      : du_dx(du_dx), du_dz(du_dz) {}
};

} // namespace point
} // namespace specfem

#endif /* _POINT_FIELD_DERIVATIVES_HPP */
