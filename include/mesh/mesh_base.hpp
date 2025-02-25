#pragma once

#include "enumerations/dimension.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Struct to store information about the mesh read from the database
 *
 */
template <specfem::dimension::type DimensionType> struct mesh;

/**
 * @brief Struct to store general parameters for the mesh
 *
 */
template <specfem::dimension::type DimensionType> struct parameters;

/**
 * @brief Struct to store materials
 *
 */
template <specfem::dimension::type DimensionType> struct materials;
/**
 * @brief Struct to store control nodes
 *
 */
template <specfem::dimension::type DimensionType> struct control_nodes;

/**
 * @brief Struct to store coupled interfaces
 *
 */
template <specfem::dimension::type DimensionType> struct coupled_interfaces;

/**
 * @brief Struct to store boundaries
 *
 */
template <specfem::dimension::type DimensionType> struct boundaries;

/**
 * @brief Struct to store absorbing boundaries
 *
 */
template <specfem::dimension::type DimensionType> struct absorbing_boundary;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 */
template <specfem::dimension::type DimensionType> struct acoustic_free_surface;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 */
template <specfem::dimension::type DimensionType> struct free_surface;

/**
 * @brief Struct to store forcing boundaries
 *
 */
template <specfem::dimension::type DimensionType> struct forcing_boundary;

/**
 * @brief Struct to store tags for every spectral element
 *
 */
template <specfem::dimension::type DimensionType> struct tags;

/**
 * @brief Struct to store mass_matrices
 *
 */
template <specfem::dimension::type DimensionType> struct mass_matrix;

/**
 * @brief Struct to store element_types (Acoustic, lastic, Poroelastic)
 *
 */
template <specfem::dimension::type DimensionType> struct element_types;

namespace elements {

template <specfem::dimension::type DimensionType> struct axial_elements;
template <specfem::dimension::type DimensionType> struct tangential_elements;

} // namespace elements

} // namespace mesh
} // namespace specfem
