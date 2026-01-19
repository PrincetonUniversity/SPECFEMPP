#pragma once

/**
 * @brief Finite element mesh data structures for SPECFEM simulations.
 *
 * Contains mesh components including element connectivity, material properties,
 * control nodes, boundary conditions, and geometric transformations. Supports
 * both 2D and 3D spectral element meshes with dimension-templated
 * specializations.
 *
 * @see specfem::mesh::mesh struct is used to store the overall mesh data as
 * members.
 */
namespace specfem::mesh {}

#include "mesh/dim2/mesh.hpp"
#include "mesh/dim3/mesh.hpp"
#include "mesh/mesh_base.hpp"
