#ifndef _TANGENTIAL_ELEMENTS_HPP
#define _TANGENTIAL_ELEMENTS_HPP

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace mesh {
namespace elements {
/**
 * Define tangential elements
 *
 * @note Need to still document this section
 *
 */
template <specfem::dimension::type DimensionTag> struct tangential_elements;

template <> struct tangential_elements<specfem::dimension::type::dim2> {

  constexpr static auto dimension = specfem::dimension::type::dim2;

  bool force_normal_to_surface, rec_normal_to_surface;
  specfem::kokkos::HostView1d<type_real> x, y;
  tangential_elements() {};
  tangential_elements(const int nnodes_tangential_curve);
};
} // namespace elements
} // namespace mesh
} // namespace specfem

#endif
