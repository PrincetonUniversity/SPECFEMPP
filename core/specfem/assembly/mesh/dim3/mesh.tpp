#pragma once

#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "mesh.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/mesh.hpp"

specfem::assembly::mesh<specfem::element::dimension_tag::dim3>::mesh(
    const int nspec, const int ngnod, const int ngllz, const int nglly,
    const int ngllx, const specfem::mesh::tags<dimension_tag> &tags,
    const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph,
    const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
    const specfem::quadrature::quadratures &quadrature)
    : nspec(nspec), element_grid(ngllz, nglly, ngllx), ngnod(ngnod) {
  // Initialize base classes
  static_cast<
      specfem::assembly::mesh_impl::mesh_to_compute_mapping<dimension_tag> &>(
      *this) = { tags, adjacency_graph };
  static_cast<specfem::assembly::mesh_impl::adjacency_graph<dimension_tag> &>(
      *this) = {
    nspec,
    static_cast<const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
        dimension_tag> &>(*this),
    adjacency_graph
  };
  static_cast<specfem::assembly::mesh_impl::control_nodes<dimension_tag> &>(
      *this) = {
    static_cast<const specfem::assembly::mesh_impl::mesh_to_compute_mapping<
        dimension_tag> &>(*this),
    control_nodes
  };
  static_cast<specfem::assembly::mesh_impl::quadrature<dimension_tag> &>(
      *this) = { quadrature };
  static_cast<specfem::assembly::mesh_impl::shape_functions<dimension_tag> &>(
      *this) = {
    ngllz,
    nglly,
    ngllx,
    ngnod,
    static_cast<
        const specfem::assembly::mesh_impl::quadrature<dimension_tag> &>(*this),
    static_cast<const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
                    &>(*this)
  };
  static_cast<specfem::assembly::mesh_impl::points<dimension_tag> &>(*this) = {
    nspec,
    ngllz,
    nglly,
    ngllx,
    static_cast<
        const specfem::assembly::mesh_impl::adjacency_graph<dimension_tag> &>(
        *this),
    static_cast<const specfem::assembly::mesh_impl::control_nodes<dimension_tag>
                    &>(*this),
    static_cast<
        const specfem::assembly::mesh_impl::shape_functions<dimension_tag> &>(
        *this)
  };

  return;
}
