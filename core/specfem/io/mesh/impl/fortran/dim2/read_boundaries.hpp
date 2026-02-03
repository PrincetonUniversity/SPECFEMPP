#pragma once

#include "specfem/mesh.hpp"

#include <fstream>
#include <vector>

namespace specfem {
namespace io {
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
//  * @return specfem::mesh::absorbing_boundary
//  */
// specfem::mesh::absorbing_boundary read_absorbing_boundaries(
//     std::ifstream &stream, const int n_absorbing, const int nspec);

// /**
//  * @brief Read acoustic free surface from mesh database
//  *
//  * @param stream Input stream
//  * @param nspec Number of spectral elements
//  * @param n_acoustic_surface Number of acoustic surfaces
//  * @return specfem::mesh::acoustic_free_surface
//  */
// specfem::mesh::acoustic_free_surface read_acoustic_free_surface(
//     std::ifstream &stream, const int nspec, const int n_acoustic_surface);

// /**
//  * @brief Read forcing boundaries from mesh database
//  *
//  * @param stream Input stream
//  * @param nspec Number of spectral elements
//  * @param n_acforcing Number of acoustic forcing boundaries
//  * @return specfem::mesh::forcing_boundary
//  */
// specfem::mesh::forcing_boundary read_forcing_boundaries(
//     std::ifstream &stream, const int nspec, const int n_acforcing);

/**
 * @brief Read boundaries from mesh database
 *
 * @param stream Input stream
 * @param nspec Number of spectral elements
 * @param n_absorbing Number of absorbing boundaries
 * @param n_acforcing Number of acoustic forcing boundaries
 * @param n_acoustic_surface Number of acoustic surfaces
 * @return specfem::mesh::boundaries
 */
specfem::mesh::boundaries<specfem::dimension::type::dim2>
read_boundaries(std::ifstream &stream, const int nspec, const int n_absorbing,
                const int n_acoustic_surface, const int n_acforcing,
                const Kokkos::View<int **, Kokkos::HostSpace> knods);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
