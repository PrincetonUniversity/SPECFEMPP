#ifndef _TANGENTIAL_ELEMENTS_HPP
#define _TANGENTIAL_ELEMENTS_HPP

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace elements {
/**
 * Define tangential elements
 *
 * @note Need to still document this section
 *
 */
struct tangential_elements {
  bool force_normal_to_surface, rec_normal_to_surface;
  specfem::kokkos::HostView1d<type_real> x, y;
  tangential_elements(){};
  tangential_elements(const int nnodes_tangential_curve);
};
} // namespace elements
} // namespace mesh
} // namespace specfem

#endif
