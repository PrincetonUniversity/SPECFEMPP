#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Struct to store information about the mesh read from the database
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct mesh;

/**
 * @brief Struct to store general parameters for the mesh
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct parameters;

/**
 * @brief Struct to store materials
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct materials;

/**
 * @brief Struct to store coordinates
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct coordinates;

/**
 * @brief Struct to store mapping
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct mapping;

/**
 * @brief Struct to store partial derivatives
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct partial_derivatives;

/**
 * @brief Struct to store control nodes
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct control_nodes;

/**
 * @brief Struct to store coupled interfaces
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct coupled_interfaces;

/**
 * @brief Struct to store boundaries
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct boundaries;

/**
 * @brief Struct to store absorbing boundaries
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct absorbing_boundary;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct acoustic_free_surface;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct free_surface;

/**
 * @brief Struct to store forcing boundaries
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct forcing_boundary;

/**
 * @brief Struct to store tags for every spectral element
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct tags;

/**
 * @brief Struct to store mass_matrices
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct mass_matrix;

/**
 * @brief Struct to store element_types (Acoustic, lastic, Poroelastic)
 *
 * @tparam DimensionType Dimension type
 *
 */
template <specfem::dimension::type DimensionType> struct element_types;

namespace elements {

template <specfem::dimension::type DimensionType> struct axial_elements;
template <specfem::dimension::type DimensionType> struct tangential_elements;

} // namespace elements

/**
 * @brief Container to store mpi information
 *
 * @tparam DimensionType
 */
template <specfem::dimension::type DimensionType> struct mpi;

/**
 * @brief Struct to store the whether elements are inner or outer MPI elements.
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct inner_outer;

/**
 * @brief Struct to store the coloring of the mesh
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct coloring;

/**
 * @brief Struct to store the surface elements
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct surface;

/**
 * @brief Struct to store the adjacency information
 *
 * @tparam DimensionType Dimension type
 */
template <specfem::dimension::type DimensionType> struct adjacency;

} // namespace mesh
} // namespace specfem
