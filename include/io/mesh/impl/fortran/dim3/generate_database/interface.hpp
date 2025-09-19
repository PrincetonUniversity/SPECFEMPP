#pragma once

#include "io/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

/**
 * @brief Read paramters from 3D mesh database
 *
 * @param stream Input stream
 * @param mpi MPI object
 * @return specfem::mesh::parameters<specfem::dimension::type::dim2> Mesh
 * parameters
 */
specfem::mesh::parameters<specfem::dimension::type::dim3>
read_mesh_parameters(std::ifstream &stream, const specfem::MPI::MPI *mpi);

/**
 * @brief Read mapping from 3D mesh database
 *
 * @param stream Input stream
 * @param mapping Mapping object
 * @param mpi MPI object
 */
void read_ibool(std::ifstream &stream,
                specfem::mesh::mapping<specfem::dimension::type::dim3> &mapping,
                const specfem::MPI::MPI *mpi);

/**
 * @brief Read element types from 3D mesh database
 *
 * @param stream Input stream
 * @param element_types Element types object
 * @param mpi MPI object
 */
void read_element_types(
    std::ifstream &stream,
    specfem::mesh::element_types<specfem::dimension::type::dim3> &element_types,
    const specfem::MPI::MPI *mpi);

/**
 * @brief Read coordinates from 3D mesh database
 *
 * @param stream Input stream
 * @param coordinates Coordinates object
 * @param mpi MPI object
 */
void read_xyz(
    std::ifstream &stream,
    specfem::mesh::coordinates<specfem::dimension::type::dim3> &coordinates,
    const specfem::MPI::MPI *mpi);

/**
 * @brief Read Jacobian from 3D mesh database
 *
 * @param stream Input stream
 * @param coordinates Coordinates object
 * @param mpi MPI object
 */
void read_jacobian_matrix(
    std::ifstream &stream,
    specfem::mesh::jacobian_matrix<specfem::dimension::type::dim3>
        &jacobian_matrix,
    const specfem::MPI::MPI *mpi);

/**
 * @brief Read array from 3D mesh database
 *
 * This function is really used to read any size array from the Fortran
 * 3D mesh database. It works with 1-5D arrays, and assumes that the first
 * dimension is always the last dimension of the fortran array. And unwraps
 * the rest of the dimensions (which are written in Fortran order) to the
 * Kokkos::View.
 *
 * @param stream Input stream
 * @param array Array to read
 * @tparam ViewType Kokkos::View type
 * @throws std::runtime_error if an error occurs while reading the array
 *
 * @code{.cpp}
 * // Example of how to use this function
 * Kokkos::View<int *, Kokkos::HostSpace> array("array", 10);
 * specfem::io::mesh::impl::fortran::dim3::read_array(stream, array);
 * @endcode
 */
template <typename ViewType>
void read_array(std::ifstream &stream, ViewType &array);

/**
 * @brief Read index array from 3D mesh database, subtracts 1 from each value
 *        to convert from Fortran to C indexing
 *
 * This function is really used to read any size index array from the Fortran
 * 3D mesh database. It works with 1-4D arrays, and assumes that the first
 * dimension is always the last dimension of the fortran array. And unwraps
 * the rest of the dimensions (which are written in Fortran order) to the
 * Kokkos::View. It also subtracts 1 from each value to convert from Fortran
 * to C indexing.
 *
 * @param stream Input stream
 * @param array Index array to read
 * @tparam ViewType Kokkos::View type
 * @throws std::runtime_error if an error occurs while reading the array
 *
 * @code{.cpp}
 * // Example of how to use this function
 * Kokkos::View<int *, Kokkos::HostSpace> array("array", 10);
 * specfem::io::mesh::impl::fortran::dim3::read_index_array(stream, array);
 * @endcode
 */
template <typename ViewType>
void read_index_array(std::ifstream &stream, ViewType &array);

/**
 * @brief Read single test value from 3D mesh database and throw error if
 *        value is not as expected
 *
 * @param stream Input stream
 * @param test_value Test value to read
 * @throws std::runtime_error if the test value is not as expected
 * @code{.cpp}
 * // Example of how to use this function
 * int test_value;
 * specfem::io::mesh::impl::fortran::dim3::check_read_test_value(stream,
 * test_value);
 * @endcode
 */
void check_read_test_value(std::ifstream &stream, int test_value);

/**
 * @brief Check values and throw error if they are not as expected
 *
 * @param message Message to print if values are not as expected
 * @param value Value to check
 * @param expected Expected value
 *
 * @throws std::runtime_error if the value is not as expected
 *
 * @code{.cpp}
 * // Example of how to use this function
 * specfem::io::mesh::impl::fortran::dim3::check_values("message", value,
 * expected);
 * @endcode
 *
 */
void check_values(std::string message, int value, int expected);

template <typename ViewType>
void read_control_nodes_indexing(std::ifstream &stream,
                                 ViewType &control_nodes_indexing);

template <typename ViewType>
void read_control_nodes_coordinates(std::ifstream &stream,
                                    ViewType &control_nodes_coordinates);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem

#include "io/mesh/impl/fortran/dim3/generate_database/interface.tpp"
#include "io/mesh/impl/fortran/dim3/generate_database/utilities.hpp"
