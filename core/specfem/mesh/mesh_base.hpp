#pragma once

#include "specfem/element.hpp"
#include "specfem/simulation.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Raw mesh data read from a model-specific mesh database.
 *
 * The raw mesh is keyed by simulation model rather than only by dimension so
 * model-specific database payloads, such as Cartesian UTM metadata or Globe3D
 * reference geometry and evaluator context, can live beside the common mesh
 * fields without runtime optionals in every mesh instance.
 *
 * @tparam ModelTag Simulation model represented by this raw mesh
 */
template <specfem::simulation::model ModelTag> struct mesh;

/** @brief Raw mesh read from a 2-D Cartesian SPECFEM database. */
using cartesian2d_mesh = mesh<specfem::simulation::model::Cartesian2D>;

/** @brief Raw mesh read from a 3-D Cartesian SPECFEM database. */
using cartesian3d_mesh = mesh<specfem::simulation::model::Cartesian3D>;

/** @brief Raw mesh read from a thin SPECFEM3D_GLOBE database. */
using globe3d_mesh = mesh<specfem::simulation::model::Globe3D>;

/**
 * @brief Struct to store general parameters for the mesh
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct parameters;

/**
 * @brief Struct to store materials
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct materials;

/**
 * @brief Struct to store coordinates
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct coordinates;

/**
 * @brief Struct to store mapping
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct mapping;

/**
 * @brief Struct to store Jacobian matrix
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct jacobian_matrix;

/**
 * @brief Struct to store control nodes
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct control_nodes;

/**
 * @brief Struct to store coupled interfaces
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag>
struct coupled_interfaces;

/**
 * @brief Struct to store boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct boundaries;

/**
 * @brief Struct to store absorbing boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag>
struct absorbing_boundary;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag>
struct acoustic_free_surface;

/**
 * @brief Struct to store acoustic free surface boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag>
struct acoustic_free_surface;

/**
 * @brief Struct to store forcing boundaries
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct forcing_boundary;

/**
 * @brief Struct to store tags for every spectral element
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct tags;

/**
 * @brief Struct to store mass_matrices
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct mass_matrix;

/**
 * @brief Struct to store element_types (Acoustic, Elastic, Poroelastic, etc.)
 *
 * @tparam DimensionTag Dimension type
 *
 */
template <specfem::element::dimension_tag DimensionTag> struct element_types;

namespace elements {

template <specfem::element::dimension_tag DimensionTag> struct axial_elements;
template <specfem::element::dimension_tag DimensionTag>
struct tangential_elements;

} // namespace elements

/**
 * @brief Container to store mpi information
 *
 * @tparam DimensionTag
 */
template <specfem::element::dimension_tag DimensionTag> struct mpi;

/**
 * @brief Struct to store the whether elements are inner or outer MPI elements.
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct inner_outer;

/**
 * @brief Struct to store the coloring of the mesh
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct coloring;

/**
 * @brief Struct to store the surface elements
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct surface;

/**
 * @brief Struct to store the adjacency information
 *
 * @tparam DimensionTag Dimension type
 */
template <specfem::element::dimension_tag DimensionTag> struct adjacency;

} // namespace mesh
} // namespace specfem
