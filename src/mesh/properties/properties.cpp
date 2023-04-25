#include "mesh/properties/properties.hpp"
#include "fortran_IO.h"

specfem::mesh::properties::properties(std::ifstream &stream,
                                      const specfem::MPI::MPI *mpi) {
  // ---------------------------------------------------------------------
  // reading mesh properties

  specfem::fortran_IO::fortran_read_line(stream, &this->numat, &this->ngnod,
                                         &this->nspec, &this->pointsdisp,
                                         &this->plot_lowerleft_corner_only);

  specfem::fortran_IO::fortran_read_line(
      stream, &this->nelemabs, &this->nelem_acforcing,
      &this->nelem_acoustic_surface, &this->num_fluid_solid_edges,
      &this->num_fluid_poro_edges, &this->num_solid_poro_edges,
      &this->nnodes_tangential_curve, &this->nelem_on_the_axis);
  // ----------------------------------------------------------------------

  mpi->sync_all();
}
