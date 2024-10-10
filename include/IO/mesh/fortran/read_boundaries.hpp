#pragma once

#include "mesh/boundaries/boundaries.hpp"
#include "specfem_mpi/interface.hpp"
#include "mesh/boundaries/absorbing_boundaries.hpp"
#include "mesh/boundaries/acoustic_free_surface.hpp"
#include "mesh/boundaries/forcing_boundaries.hpp"
#include <fstream>
#include <vector>


namespace specfem {
namespace IO {
namespace mesh {
namespace fortran {

    /**
     * @brief Read absorbing boundaries from mesh database
     *
     * @param stream Input stream
     * @param nspec Number of spectral elements
     * @param n_absorbing Number of absorbing boundaries
     * @param mpi MPI object
     * @return specfem::mesh::absorbing_boundary
     */
    specfem::mesh::absorbing_boundary read_absorbing_boundaries(
        std::ifstream &stream, const int n_absorbing, const int nspec, 
        const specfem::MPI::MPI *mpi) {};

    /**
     * @brief Read acoustic free surface from mesh database
     *
     * @param stream Input stream
     * @param nspec Number of spectral elements
     * @param n_acoustic_surface Number of acoustic surfaces
     * @param mpi MPI object
     * @return specfem::mesh::acoustic_free_surface
     */
    specfem::mesh::acoustic_free_surface read_acoustic_free_surface(
        std::ifstream &stream, const int nspec, const int n_acoustic_surface, 
        const specfem::MPI::MPI *mpi) {};

    /**
     * @brief Read forcing boundaries from mesh database
     *
     * @param stream Input stream
     * @param nspec Number of spectral elements
     * @param n_acforcing Number of acoustic forcing boundaries
     * @param mpi MPI object
     * @return specfem::mesh::forcing_boundary
     */
    specfem::mesh::forcing_boundary read_forcing_boundaries(
        std::ifstream &stream, const int nspec, const int n_acforcing, 
        const specfem::MPI::MPI *mpi) {};

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
    specfem::mesh::boundaries read_boundaries(
        std::ifstream &stream, const int nspec, const int n_absorbing, 
        const int n_acforcing, const int n_acoustic_surface, 
        const specfem::MPI::MPI *mpi) {};

} // namespace fortran
} // namespace mesh
} // namespace IO
} // namespace specfem
