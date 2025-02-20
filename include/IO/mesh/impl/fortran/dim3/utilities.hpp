#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>

template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

template <typename T> using View4D = Kokkos::View<T ****, Kokkos::HostSpace>;

template <typename T> using View5D = Kokkos::View<T *****, Kokkos::HostSpace>;

template <typename T>
void specfem::IO::mesh::impl::fortran::dim3::read_array(std::ifstream &stream,
                                                        View1D<T> &array) {
  const int n = array.extent(0);
  std::vector<T> dummy_T(n, -999999);

  try {
    // Read into dummy vector
    specfem::IO::fortran_read_line(stream, &dummy_T);
    // Assign to KokkosView
    for (int i = 0; i < n; i++) {
      array(i) = dummy_T[i];
    }
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading array from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

template <typename T>
void specfem::IO::mesh::impl::fortran::dim3::read_array(std::ifstream &stream,
                                                        View4D<T> &array) {
  const int nspec = array.extent(0);
  const int ngllx = array.extent(1);
  const int nglly = array.extent(2);
  const int ngllz = array.extent(3);

  std::vector<T> dummy_T(ngllx * nglly * ngllz, -999999);

  try {
    for (int ispec = 0; ispec < nspec; ispec++) {
      // Read into dummy vector for each ispec
      specfem::IO::fortran_read_line(stream, &dummy_T);

      // Assign to KokkosView
      int counter = 0;
      for (int igllx = 0; igllx < ngllx; igllx++) {
        for (int iglly = 0; iglly < nglly; iglly++) {
          for (int igllz = 0; igllz < ngllz; igllz++) {
            array(ispec, igllx, iglly, igllz) = dummy_T[counter];
            counter++;
          }
        }
      }
    }

  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading array from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

template <typename T>
void specfem::IO::mesh::impl::fortran::dim3::read_array(std::ifstream &stream,
                                                        View5D<T> &array) {
  const int nspec = array.extent(0);
  const int ngllx = array.extent(1);
  const int nglly = array.extent(2);
  const int ngllz = array.extent(3);
  const int ncomp = array.extent(4);

  std::vector<T> dummy_T(ngllx * nglly * ngllz * ncomp, -9999999);

  try {
    for (int ispec = 0; ispec < nspec; ispec++) {
      // Read into dummy vector for each ispec
      specfem::IO::fortran_read_line(stream, &dummy_T);

      // Assign to KokkosView
      int counter = 0;
      for (int igllx = 0; igllx < ngllx; igllx++) {
        for (int iglly = 0; iglly < nglly; iglly++) {
          for (int igllz = 0; igllz < ngllz; igllz++) {
            for (int icomp = 0; icomp < ncomp; icomp++) {
              array(ispec, igllx, iglly, igllz, icomp) = dummy_T[counter];
              counter++;
            }
          }
        }
      }
    }
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading array from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

template <typename T>
void specfem::IO::mesh::impl::fortran::dim3::read_index_array(
    std::ifstream &stream, View1D<T> &array) {
  const int n = array.extent(0);
  std::vector<T> dummy_T(n, -999999);

  try {
    // Read into dummy vector
    specfem::IO::fortran_read_line(stream, &dummy_T);

    // Assign to KokkosView
    for (int i = 0; i < n; i++) {
      array(i) = dummy_T[i] - 1;
    }

  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading 1D index_array from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}

template <typename T>
void specfem::IO::mesh::impl::fortran::dim3::read_index_array(
    std::ifstream &stream, View4D<T> &array) {
  const int nspec = array.extent(0);
  const int ngllx = array.extent(1);
  const int nglly = array.extent(2);
  const int ngllz = array.extent(3);

  std::vector<T> dummy_T(ngllx * nglly * ngllz, -9999.0);

  try {
    for (int ispec = 0; ispec < nspec; ispec++) {
      // Read into dummy vector for each ispec
      specfem::IO::fortran_read_line(stream, &dummy_T);

      // Assign to KokkosView
      int counter = 0;
      for (int igllx = 0; igllx < ngllx; igllx++) {
        for (int iglly = 0; iglly < nglly; iglly++) {
          for (int igllz = 0; igllz < ngllz; igllz++) {
            array(ispec, igllx, iglly, igllz) = dummy_T[counter] - 1;
            counter++;
          }
        }
      }
    }

  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message << "Error reading array from database file:\n"
                  << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }
}
