#pragma once

#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::mesh_impl {
template <specfem::dimension::type Dimension> struct control_nodes;

template <specfem::dimension::type Dimension> struct mesh_to_compute_mapping;

template <specfem::dimension::type Dimension> struct points;

template <specfem::dimension::type Dimension> struct shape_functions;

template <specfem::dimension::type Dimension> struct quadrature;
} // namespace specfem::assembly::mesh_impl

namespace specfem::assembly {
template <specfem::dimension::type Dimension> struct mesh;
} // namespace specfem::assembly

// Include dim2 declarations
#include "mesh/dim2/mesh.hpp"
// Include dim3 declarations
#include "mesh/dim3/mesh.hpp"
