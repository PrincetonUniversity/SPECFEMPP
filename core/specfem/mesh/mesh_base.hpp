#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Struct to store information about the mesh read from the database
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct mesh;

/**
 * @brief Struct to store general parameters for the mesh
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct parameters;

/**
 * @brief Struct to store materials
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct materials;

/**
 * @brief Struct to store coordinates
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct coordinates;

/**
 * @brief Struct to store mapping
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct mapping;

/**
 * @brief Struct to store Jacobian matrix
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct jacobian_matrix;

/**
 * @brief Struct to store control nodes
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct control_nodes;

/**
 * @brief Struct to store coupled interfaces
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct coupled_interfaces;

/**
 * @brief Struct to store boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct boundaries;

/**
 * @brief Struct to store absorbing boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct absorbing_boundary;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct acoustic_free_surface;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct acoustic_free_surface;

/**
 * @brief Struct to store forcing boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct forcing_boundary;

/**
 * @brief Struct to store tags for every spectral element
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct tags;

/**
 * @brief Struct to store mass_matrices
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct mass_matrix;

/**
 * @brief Struct to store element_types (Acoustic, Elastic, Poroelastic, etc.)
 *
 * @tparam DimensionTag Dimension type
 *
 */
template <specfem::dimension::type DimensionTag> struct element_types;

namespace elements {

template <specfem::dimension::type DimensionTag> struct axial_elements;
template <specfem::dimension::type DimensionTag> struct tangential_elements;

} // namespace elements

/**
 * @brief Container to store mpi information
 *
 * @tparam DimensionTag
 */
template <specfem::dimension::type DimensionTag> struct mpi;

/**
 * @brief Struct to store the whether elements are inner or outer MPI elements.
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct inner_outer;

/**
 * @brief Struct to store the coloring of the mesh
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct coloring;

/**
 * @brief Struct to store the surface elements
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct surface;

/**
 * @brief Struct to store the adjacency information
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::dimension::type DimensionTag> struct adjacency;

} // namespace mesh
} // namespace specfem
