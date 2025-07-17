#pragma once

#include "enumerations/interface.hpp"
#include "boundary_values.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/element_types.hpp"

// Explicit template instantiations

template <specfem::dimension::type DimensionTag>
specfem::assembly::boundary_values<DimensionTag>::boundary_values(
    const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries)
    : stacey(nstep, mesh, element_types, boundaries),
      composite_stacey_dirichlet(nstep, mesh, element_types, boundaries) {}
