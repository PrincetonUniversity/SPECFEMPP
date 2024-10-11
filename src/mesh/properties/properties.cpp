#include "mesh/properties/properties.hpp"
#include "IO/fortranio/interface.hpp"

specfem::mesh::properties::properties(std::ifstream &stream,
                                      const specfem::MPI::MPI *mpi) {
  // ---------------------------------------------------------------------
  // reading mesh properties

  specfem::IO::fortran_read_line(stream, &this->numat, &this->ngnod,
                                 &this->nspec, &this->pointsdisp,
                                 &this->plot_lowerleft_corner_only);

  // ---------------------------------------------------------------------
  if (this->ngnod != 9) {
    std::ostringstream error_message;
    error_message << "Number of control nodes per element must be 9, but is "
                  << this->ngnod << "\n"
                  << "Currently, there is a bug when NGNOD == 4 \n";
    throw std::runtime_error(error_message.str());
  }

  specfem::IO::fortran_read_line(
      stream, &this->nelemabs, &this->nelem_acforcing,
      &this->nelem_acoustic_surface, &this->num_fluid_solid_edges,
      &this->num_fluid_poro_edges, &this->num_solid_poro_edges,
      &this->nnodes_tangential_curve, &this->nelem_on_the_axis);
  // ----------------------------------------------------------------------

  mpi->sync_all();
}
