#ifndef ELEMENTS_H
#define ELEMENTS_H

#include "../include/kokkos_abstractions.h"
#include "../include/specfem_mpi.h"

namespace specfem {

/**
 * Store data defining to specific element types
 *
 */
namespace elements {

/**
 * Define tangential elements
 *
 * @note Need to still document this section
 *
 */
struct tangential_elements {
  bool force_normal_to_surface, rec_normal_to_surface;
  specfem::HostView1d<type_real> x, y;
  tangential_elements(){};
  tangential_elements(const int nnodes_tangential_curve);
  tangential_elements(std::ifstream &stream, const int nnodes_tangential_curve);
};

/**
 * Define axial elements
 *
 * @note Need to still document this section
 *
 */
struct axial_elements {
  specfem::HostView1d<bool> is_on_the_axis;
  axial_elements(){};
  axial_elements(const int nspec);
  axial_elements(std::ifstream &stream, const int nelem_on_the_axis,
                 const int nspec, const specfem::MPI::MPI *mpi);
};

} // namespace elements
} // namespace specfem

#endif
