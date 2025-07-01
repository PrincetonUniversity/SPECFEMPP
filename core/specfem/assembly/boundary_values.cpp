#include "boundary_values.hpp"
#include "boundaries.hpp"
#include "boundary_values/boundary_medium_container.tpp"
#include "boundary_values/boundary_values_container.tpp"
#include "mesh.hpp"
#include "properties.hpp"

// Explicit template instantiations

template class specfem::assembly::impl::boundary_medium_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::boundary_tag::stacey>;

template class specfem::assembly::impl::boundary_medium_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    specfem::element::boundary_tag::stacey>;

template class specfem::assembly::boundary_value_container<
    specfem::dimension::type::dim2, specfem::element::boundary_tag::stacey>;

specfem::assembly::boundary_values::boundary_values(
    const int nstep, const specfem::assembly::mesh mesh,
    const specfem::assembly::element_types element_types,
    const specfem::assembly::boundaries boundaries)
    : stacey(nstep, mesh, element_types, boundaries),
      composite_stacey_dirichlet(nstep, mesh, element_types, boundaries) {}
