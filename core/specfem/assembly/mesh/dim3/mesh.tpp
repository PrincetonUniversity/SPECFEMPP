#pragma once

#include "mesh.hpp"
#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "mesh/mesh.hpp"

specfem::assembly::mesh<specfem::dimension::type::dim3>::mesh(
    const specfem::mesh::parameters<dimension_tag> &parameters,
    const specfem::mesh::coordinates<dimension_tag> &coordinates,
    const specfem::mesh::mapping<dimension_tag> &mapping,
    const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
    const specfem::quadrature::quadratures &quadrature)
    : nspec(parameters.nspec), ngllz(parameters.ngllz), nglly(parameters.nglly),
      ngllx(parameters.ngllx),
      ngnod(parameters.ngnod), specfem::assembly::mesh_impl::points<dimension_tag>(mapping, coordinates),
       specfem::assembly::mesh_impl::quadrature<dimension_tag>(quadrature),
       specfem::assembly::mesh_impl::control_nodes<dimension_tag>(control_nodes) {}
