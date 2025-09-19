#pragma once

#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename ViewType>
void specfem::io::mesh::impl::fortran::dim3::read_array(std::ifstream &stream,
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
      specfem::io::fortran_read_line(stream, &dummy_T);
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
        specfem::io::fortran_read_line(stream, &dummy_T);
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
        specfem::io::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int k = 0; k < n2; k++) {
          for (int j = 0; j < n1; j++) {
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
        specfem::io::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int l = 0; l < n3; l++) {
          for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n1; j++) {
              array(i, l, k, j) = dummy_T[counter];
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
        specfem::io::fortran_read_line(stream, &dummy_T);
        // Assign to KokkosView
        int counter = 0;
        for (int m = 0; m < n4; m++) {
          for (int l = 0; l < n3; l++) {
            for (int k = 0; k < n2; k++) {
              for (int j = 0; j < n1; j++) {
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
void specfem::io::mesh::impl::fortran::dim3::read_index_array(
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
      specfem::io::fortran_read_line(stream, &dummy_T);

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
        specfem::io::fortran_read_line(stream, &dummy_T);

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
        specfem::io::fortran_read_line(stream, &dummy_T);

        // Assign to KokkosView
        int counter = 0;
        for (int k = 0; k < n2; k++) {
          for (int j = 0; j < n1; j++) {
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
        specfem::io::fortran_read_line(stream, &dummy_T);

        // Assign to KokkosView
        int counter = 0;
        for (int l = 0; l < n3; l++) {
          for (int k = 0; k < n2; k++) {
            for (int j = 0; j < n1; j++) {
              array(i, l, k, j) = dummy_T[counter] - 1;
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

template <typename ViewType>
void specfem::io::mesh::impl::fortran::dim3::read_control_nodes_indexing(
    std::ifstream &stream, ViewType &control_nodes_indexing) {
  // Get value_type and rank of the ViewType
  using value_type = typename ViewType::value_type;
  constexpr int rank = ViewType::rank;

  if (rank != 2) {
    throw std::runtime_error("Rank for control nodes index must be 2");
  }

  const int nspec = control_nodes_indexing.extent(0);
  const int ngnod = control_nodes_indexing.extent(1);

  std::vector<value_type> dummy_T(ngnod, -999999);

  try {
    for (int i = 0; i < nspec; i++) {
      // Read into dummy vector
      specfem::io::fortran_read_line(stream, &dummy_T);

      // Assign to KokkosView
      int counter = 0;
      for (int j = 0; j < ngnod; j++) {
        control_nodes_indexing(i, j) = dummy_T[counter] - 1;
        counter++;
      }
    }
  } catch (std::runtime_error &e) {
    std::ostringstream error_message;
    error_message
        << "Error reading control nodes indexing from database file:\n"
        << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
    throw std::runtime_error(error_message.str());
  }

  return;
}

template <typename ViewType>
void specfem::io::mesh::impl::fortran::dim3::read_control_nodes_coordinates(
    std::ifstream &stream, ViewType &control_nodes_coordinates) {
  // Get value_type and rank of the ViewType
  using value_type = typename ViewType::value_type;
  constexpr int rank = ViewType::rank;

  if (rank != 2) {
    throw std::runtime_error("Rank for control nodes coordinates must be 2");
  }

  const int nnodes = control_nodes_coordinates.extent(0);
  const int ndim = control_nodes_coordinates.extent(1);

  for (int index = 0; index < nnodes; index++) {
    std::vector<value_type> dummy_T(3, -999999);
    try {
      // Read into dummy vector
      specfem::io::fortran_read_line(stream, &dummy_T);
      // Assign to KokkosView
      for (int k = 0; k < ndim; k++) {
        control_nodes_coordinates(index, k) = dummy_T[k];
      }
    } catch (std::runtime_error &e) {
      std::ostringstream error_message;
      error_message
          << "Error reading control nodes coordinates from database file:\n"
          << e.what() << "(" << __FILE__ << ":" << __LINE__ << ")";
      throw std::runtime_error(error_message.str());
    }
  }

  return;
}
