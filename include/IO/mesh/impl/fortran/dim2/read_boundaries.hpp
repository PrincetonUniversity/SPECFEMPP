#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>
#include <vector>

namespace specfem {
namespace IO {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

// /**
//  * @brief Read absorbing boundaries from mesh database
//  *
//  * @param stream Input stream
//  * @param nspec Number of spectral elements
//  * @param n_absorbing Number of absorbing boundaries
//  * @param mpi MPI object
//  * @return specfem::mesh::absorbing_boundary
//  */
// specfem::mesh::absorbing_boundary read_absorbing_boundaries(
//     std::ifstream &stream, const int n_absorbing, const int nspec,
//     const specfem::MPI::MPI *mpi);

// /**
//  * @brief Read acoustic free surface from mesh database
//  *
//  * @param stream Input stream
//  * @param nspec Number of spectral elements
//  * @param n_acoustic_surface Number of acoustic surfaces
//  * @param mpi MPI object
//  * @return specfem::mesh::acoustic_free_surface
//  */
// specfem::mesh::acoustic_free_surface read_acoustic_free_surface(
//     std::ifstream &stream, const int nspec, const int n_acoustic_surface,
//     const specfem::MPI::MPI *mpi);

// /**
//  * @brief Read forcing boundaries from mesh database
//  *
//  * @param stream Input stream
//  * @param nspec Number of spectral elements
//  * @param n_acforcing Number of acoustic forcing boundaries
//  * @param mpi MPI object
//  * @return specfem::mesh::forcing_boundary
//  */
// specfem::mesh::forcing_boundary read_forcing_boundaries(
//     std::ifstream &stream, const int nspec, const int n_acforcing,
//     const specfem::MPI::MPI *mpi);

/**
 * @brief Read boundaries from mesh database
 *
 * @param stream Input stream
 * @param nspec Number of spectral elements
 * @param n_absorbing Number of absorbing boundaries
 * @param n_acforcing Number of acoustic forcing boundaries
 * @param n_acoustic_surface Number of acoustic surfaces
 * @param mpi MPI object
 * @return specfem::mesh::boundaries
 */
specfem::mesh::boundaries<specfem::dimension::type::dim2>
read_boundaries(std::ifstream &stream, const int nspec, const int n_absorbing,
                const int n_acoustic_surface, const int n_acforcing,
                const Kokkos::View<int **, Kokkos::HostSpace> knods,
                const specfem::MPI::MPI *mpi);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace IO
} // namespace specfem
