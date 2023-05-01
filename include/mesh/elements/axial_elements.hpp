#ifndef _AXIAL_ELEMENTS_HPP
#define _AXIAL_ELEMENTS_HPP

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace elements {
/**
 * Define axial elements
 *
 * @note Need to still document this section
 *
 */
struct axial_elements {
  specfem::kokkos::HostView1d<bool> is_on_the_axis;
  axial_elements(){};
  axial_elements(const int nspec);
  axial_elements(std::ifstream &stream, const int nelem_on_the_axis,
                 const int nspec, const specfem::MPI::MPI *mpi);
};
} // namespace elements
} // namespace mesh
} // namespace specfem

#endif
