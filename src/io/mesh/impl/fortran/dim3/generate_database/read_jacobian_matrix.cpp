#include "enumerations/dimension.hpp"
#include "io/fortranio/interface.hpp"
#include "io/mesh/impl/fortran/dim3/generate_database/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>

// temolate instationation
// using View4D = specfem::io::mesh::impl::fortran::dim3::View4D<type_real>;
// template<> void
// specfem::io::mesh::impl::fortran::dim3::read_array<type_real>;

void specfem::io::mesh::impl::fortran::dim3::read_jacobian_matrix(
    std::ifstream &stream,
    specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    const specfem::MPI::MPI *mpi) {

  // Read Jacobian matrix
  const int nspec = jacobian_matrix.nspec;
  const int ngllx = jacobian_matrix.ngllx;
  const int nglly = jacobian_matrix.nglly;
  const int ngllz = jacobian_matrix.ngllz;

  // Init line reading dummy variable

  std::vector<type_real> dummy_tr(ngllx * nglly * ngllz, -9999.0);
  std::cout << " size of type_real: " << sizeof(type_real) << std::endl;

  // print nspec, ngllx, nglly, ngllz
  std::cout << "nspec: " << nspec << std::endl;
  std::cout << "ngllx: " << ngllx << std::endl;
  std::cout << "nglly: " << nglly << std::endl;
  std::cout << "ngllz: " << ngllz << std::endl;

  // Read all elements at once
  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.xix);

  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xix from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.xiy);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xiy from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.xiz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xiz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.etax);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etax from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.etay);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etay from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.etaz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etaz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.gammax);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammax from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.gammay);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammay from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(stream,
                                                       jacobian_matrix.gammaz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammaz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::io::mesh::impl::fortran::dim3::read_array(
        stream, jacobian_matrix.jacobian);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading jacobian from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}
