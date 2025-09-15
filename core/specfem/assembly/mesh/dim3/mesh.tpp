#pragma once

#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "mesh.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"

specfem::assembly::mesh<specfem::dimension::type::dim3>::mesh(
    const specfem::mesh::parameters<dimension_tag> &parameters,
    const specfem::mesh::coordinates<dimension_tag> &coordinates,
    const specfem::mesh::mapping<dimension_tag> &mapping,
    const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
    const specfem::quadrature::quadratures &quadrature)
    : nspec(parameters.nspec),
      element_grid(parameters.ngllz, parameters.nglly, parameters.ngllx),
      ngnod(parameters.ngnod),
      specfem::assembly::mesh_impl::points<dimension_tag>(mapping, coordinates),
      specfem::assembly::mesh_impl::quadrature<dimension_tag>(quadrature),
      specfem::assembly::mesh_impl::control_nodes<dimension_tag>(control_nodes),
      specfem::assembly::mesh_impl::shape_functions<dimension_tag>(
          quadrature.gll.get_hxi(), quadrature.gll.get_hxi(),
          quadrature.gll.get_hxi(), parameters.ngllz, parameters.ngnod) {}
