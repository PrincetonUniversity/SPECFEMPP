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

template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

template <typename T> using View4D = Kokkos::View<T ****, Kokkos::HostSpace>;

template <typename T> using View5D = Kokkos::View<T *****, Kokkos::HostSpace>;

template <typename T> void read_array(std::ifstream &stream, View1D<T> &array);

template <typename T> void read_array(std::ifstream &stream, View4D<T> &array);

template <typename T> void read_array(std::ifstream &stream, View5D<T> &array);

// Read index array will subtract 1 from each value when reading to account for
// Fortran 1-based indexing
template <typename T>
void read_index_array(std::ifstream &stream, View1D<T> &array);

template <typename T>
void read_index_array(std::ifstream &stream, View4D<T> &array);

} // namespace dim3
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem

#include "IO/mesh/impl/fortran/dim3/utilities.hpp"
