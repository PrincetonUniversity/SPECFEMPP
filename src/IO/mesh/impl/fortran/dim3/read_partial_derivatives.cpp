#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>

// temolate instationation
// using View4D = specfem::IO::mesh::impl::fortran::dim3::View4D<type_real>;
// template<> void
// specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>;

void specfem::IO::mesh::impl::fortran::dim3::read_partial_derivatives(
    std::ifstream &stream,
    specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>
        &partial_derivatives,
    const specfem::MPI::MPI *mpi) {

  // Read partial derivatives
  const int nspec = partial_derivatives.nspec;
  const int ngllx = partial_derivatives.ngllx;
  const int nglly = partial_derivatives.nglly;
  const int ngllz = partial_derivatives.ngllz;

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
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.xix);

  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xix from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.xiy);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xiy from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.xiz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading xiz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.etax);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etax from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.etay);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etay from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.etaz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading etaz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.gammax);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammax from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.gammay);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammay from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  try {
    specfem::IO::mesh::impl::fortran::dim3::read_array<type_real>(
        stream, partial_derivatives.gammaz);
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading gammaz from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}
