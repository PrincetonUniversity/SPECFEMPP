#include "IO/mesh/impl/fortran/dim3/read_parameters.hpp"
#include "IO/fortranio/interface.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/parameters/parameters.hpp"
#include "specfem_mpi/interface.hpp"

specfem::mesh::parameters<specfem::dimension::type::dim3>
specfem::IO::mesh::impl::fortran::dim3::read_mesh_parameters(
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  // Initialize test parameter
  int itest = -9999;

  // Instantiate parameters object
  specfem::mesh::parameters<specfem::dimension::type::dim3> mesh_parameters;

  // Read testparamater
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9999) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  }

  // Read parameters
  specfem::IO::fortran_read_line(
      stream, &mesh_parameters.acoustic_simulation,
      &mesh_parameters.elastic_simulation,
      &mesh_parameters.poroelastic_simulation, &mesh_parameters.anisotropy,
      &mesh_parameters.stacey_abc, &mesh_parameters.pml_abc,
      &mesh_parameters.approximate_ocean_load,
      &mesh_parameters.use_mesh_coloring);

  // Read test parameter
  specfem::IO::fortran_read_line(stream, &itest);

  // Throw error if test parameter is not correctly read
  if (itest != 9998) {
    std::ostringstream error_message;
    error_message << "Error reading mesh parameters: " << itest
                  << " // FILE:LINE " << __FILE__ << ":" << __LINE__ << "\n";
    throw std::runtime_error(error_message.str());
  };

  mpi->sync_all();

  return mesh_parameters;
}
