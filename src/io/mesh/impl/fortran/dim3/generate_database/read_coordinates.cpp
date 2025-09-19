#include "io/mesh/impl/fortran/dim3/generate_database/interface.hpp"
#include "specfem_setup.hpp"

void specfem::io::mesh::impl::fortran::dim3::read_xyz(
    std::ifstream &stream,
    specfem::mesh::coordinates<specfem::dimension::type::dim3> &coordinates,
    const specfem::MPI::MPI *mpi) {

  std::vector<type_real> dummy_f(coordinates.nglob, -9999.0);

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.x);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading x from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.y);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading y from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream, coordinates.z);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading z from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}
