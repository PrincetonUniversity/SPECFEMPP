#pragma once

#include "IO/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim3 {

/*
 * @brief Read paramters from 3D mesh database
 *
 * @param stream Input stream
 * @param mpi MPI object
 * @return specfem::mesh::parameters<specfem::dimension::type::dim2> Mesh
 * parameters
 */
specfem::mesh::parameters<specfem::dimension::type::dim3>
read_mesh_parameters(std::ifstream &stream, const specfem::MPI::MPI *mpi);

/*
 * @brief Read mapping from 3D mesh database
 *
 * @param stream Input stream
 * @param mapping Mapping object
 * @param mpi MPI object
 */
void read_ibool(std::ifstream &stream,
                specfem::mesh::mapping<specfem::dimension::type::dim3> &mapping,
                const specfem::MPI::MPI *mpi);

/*
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

/*
 * @brief Read Jacobian from 3D mesh database
 *
 * @param stream Input stream
 * @param coordinates Coordinates object
 * @param mpi MPI object
 */
void read_partial_derivatives(
    std::ifstream &stream,
    specfem::mesh::partial_derivatives<specfem::dimension::type::dim3>
        &partial_derivatives,
    const specfem::MPI::MPI *mpi);

/*
 * @brief Read array from 3D mesh database
 *
 * @param stream Input stream
 * @param array Array to read
 */
template <typename ViewType>
void read_array(std::ifstream &stream, ViewType &array);

/*
 * @brief Read index array from 3D mesh database, subtracts 1 from each value
 *        to convert from Fortran to C indexing
 *
 * @param stream Input stream
 * @param array Index array to read
 */
template <typename ViewType>
void read_index_array(std::ifstream &stream, ViewType &array);

/*
 * @brief Read single test value from 3D mesh database and throw error if
 *        value is not as expected
 *
 * @param stream Input stream
 * @param test_value Test value to read
 */
void check_read_test_value(std::ifstream &stream, int test_value);

/*
 * @brief Check values and throw error if they are not as expected
 *
 * @param message Message to print if values are not as expected
 * @param value Value to check
 * @param expected Expected value
 */
void check_values(std::string message, int value, int expected);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem

#include "IO/mesh/impl/fortran/dim3/utilities.hpp"
