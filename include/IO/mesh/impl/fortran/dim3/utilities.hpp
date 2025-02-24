#pragma once
#include "IO/fortranio/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/interface.hpp"
#include "IO/mesh/impl/fortran/dim3/utilities.hpp"
#include "enumerations/dimension.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

template <typename ViewType>
void specfem::IO::mesh::impl::fortran::dim3::read_array(std::ifstream &stream,
                                                        ViewType &array) {

  // Get the value_type and rank of the ViewType
  using value_type = typename ViewType::value_type;
  constexpr int rank = ViewType::rank;

  // 1D array implementation
  if constexpr (rank == 1) {

    const int n = array.extent(0);

    std::vector<value_type> dummy_T(n, -99999);

    try {
      // Read into dummy vector
      specfem::IO::fortran_read_line(stream, &dummy_T);
      // Assign to KokkosView
      for (int i = 0; i < n; i++) {
        array(i) = dummy_T[i];
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 1D array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 2D array implementation
  else if constexpr (rank == 2) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);

    std::vector<value_type> dummy_T(n1, -99999);

    try {
      // Read into dummy vector
      // Assign to KokkosView
      for (int i = 0; i < n0; i++) {
        specfem::IO::fortran_read_line(stream, &dummy_T);
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          array(i, j) = dummy_T[counter];
          counter++;
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 2D array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 3D array implementation
  else if constexpr (rank == 3) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);
    const int n2 = array.extent(2);

    std::vector<value_type> dummy_T(n1 * n2, -99999);

    try {
      // Read into dummy vector
      for (int i = 0; i < n0; i++) {
        specfem::IO::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < n2; k++) {
            array(i, j, k) = dummy_T[counter];
            counter++;
          }
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 3D array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 4D array implementation
  else if constexpr (rank == 4) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);
    const int n2 = array.extent(2);
    const int n3 = array.extent(3);

    std::vector<value_type> dummy_T(n1 * n2 * n3, -99999);

    try {
      for (int i = 0; i < n0; i++) {
        // Read into dummy vector
        specfem::IO::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < n2; k++) {
            for (int l = 0; l < n3; l++) {
              array(i, j, k, l) = dummy_T[counter];
              counter++;
            }
          }
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 4D array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 5D array implementation
  else if constexpr (rank == 5) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);
    const int n2 = array.extent(2);
    const int n3 = array.extent(3);
    const int n4 = array.extent(4);

    std::vector<value_type> dummy_T(n1 * n2 * n3 * n4, -99999);

    try {
      for (int i = 0; i < n0; i++) {
        // Read into dummy vector
        specfem::IO::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < n2; k++) {
            for (int l = 0; l < n3; l++) {
              for (int m = 0; m < n4; m++) {
                array(i, j, k, l, m) = dummy_T[counter];
                counter++;
              }
            }
          }
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 5D array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  } else {
    throw std::runtime_error("Unsupported rank for read_array array");
  }
};

template <typename ViewType>
void specfem::IO::mesh::impl::fortran::dim3::read_index_array(
    std::ifstream &stream, ViewType &array) {

  // Get value_type and rank of the ViewType
  using value_type = typename ViewType::value_type;
  constexpr int rank = ViewType::rank;

  // 1D array implementation
  if constexpr (rank == 1) {

    const int n = array.extent(0);
    std::vector<value_type> dummy_T(n, -999999);

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
  // 2D array implementation
  else if constexpr (rank == 2) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);

    std::vector<value_type> dummy_T(n1, -999999);

    try {
      for (int i = 0; i < n0; i++) {
        // Read into dummy vector
        specfem::IO::fortran_read_line(stream, &dummy_T);

        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          array(i, j) = dummy_T[counter] - 1;
          counter++;
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 2D index_array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 3D array implementation
  else if constexpr (rank == 3) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);
    const int n2 = array.extent(2);

    std::vector<value_type> dummy_T(n1 * n2, -999999);

    try {
      for (int i = 0; i < n0; i++) {
        // Read into dummy vector
        specfem::IO::fortran_read_line(stream, &dummy_T);

        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < n2; k++) {
            array(i, j, k) = dummy_T[counter] - 1;
            counter++;
          }
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 3D index_array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }
  // 4D array implementation
  else if constexpr (rank == 4) {
    const int n0 = array.extent(0);
    const int n1 = array.extent(1);
    const int n2 = array.extent(2);
    const int n3 = array.extent(3);

    std::vector<value_type> dummy_T(n1 * n2 * n3, -999999);

    try {
      for (int i = 0; i < n0; i++) {
        // Read into dummy vector
        specfem::IO::fortran_read_line(stream, &dummy_T);

        // Assign to KokkosView
        int counter = 0;
        for (int j = 0; j < n1; j++) {
          for (int k = 0; k < n2; k++) {
            for (int l = 0; l < n3; l++) {
              array(i, j, k, l) = dummy_T[counter] - 1;
              counter++;
            }
          }
        }
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message << "Error reading 4D index_array from database file:\n"
                    << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  } else {
    throw std::runtime_error("Unsupported rank for read_index_array");
  }
};

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

// Primary template for try-catch wrapper
/*
 * @brief Try-catch wrapper for fortran_read_line
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to fortran_read_line
 */
template <typename... Args>
auto try_read_line(const std::string &message, Args &&...args)
    -> decltype(specfem::IO::fortran_read_line(std::forward<Args>(args)...)) {
  try {
    return specfem::IO::fortran_read_line(std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_line(" << message
                  << ", ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in " << message;
    throw std::runtime_error(error_message.str());
  }
}

/* @brief Try-catch wrapper for read_array
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to read_array
 */
template <typename... Args>
auto try_read_array(const std::string &message, Args &&...args)
    -> decltype(specfem::IO::mesh::impl::fortran::dim3::read_array(
        std::forward<Args>(args)...)) {
  try {
    return specfem::IO::mesh::impl::fortran::dim3::read_array(
        std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_array('" << message
                  << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in try_read_array('" << message
                  << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

/* @brief Try-catch wrapper for read_index_array
 *
 * @param message Message to print if exception is caught
 * @param args Arguments to pass to read_index_array
 */
template <typename... Args>
auto try_read_index_array(const std::string &message, Args &&...args)
    -> decltype(specfem::IO::mesh::impl::fortran::dim3::read_index_array(
        std::forward<Args>(args)...)) {
  try {
    return specfem::IO::mesh::impl::fortran::dim3::read_index_array(
        std::forward<Args>(args)...);
  } catch (const std::exception &e) {
    std::ostringstream error_message;
    error_message << "Exception caught in try_read_index('" << message
                  << "', ...): " << e.what();
    throw std::runtime_error(error_message.str());
  } catch (...) {
    std::ostringstream error_message;
    error_message << "Unknown exception caught in try_read_index('" << message
                  << "', ...).";
    throw std::runtime_error(error_message.str());
  }
}

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
