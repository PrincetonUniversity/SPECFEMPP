#ifndef _AXIAL_ELEMENTS_HPP
#define _AXIAL_ELEMENTS_HPP

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"

namespace specfem {
namespace mesh {
namespace elements {
/**
 * Define axial elements
 *
 * @note Need to still document this section
 *
 */
template <specfem::dimension::type DimensionTag> struct axial_elements;

template <> struct axial_elements<specfem::dimension::type::dim2> {

  constexpr static auto dimension = specfem::dimension::type::dim2;

  specfem::kokkos::HostView1d<bool> is_on_the_axis;

  axial_elements() {};
  axial_elements(const int nspec);
  axial_elements(std::ifstream &stream, const int nelem_on_the_axis,
                 const int nspec);
};
} // namespace elements
} // namespace mesh
} // namespace specfem

#endif
