#include "compute/boundaries/boundaries.hpp"
#include "compute/boundary_values/boundary_values.hpp"
#include "compute/boundary_values/boundary_values_container.tpp"
#include "compute/boundary_values/impl/boundary_medium_container.tpp"
#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"

// Explicit template instantiations

template class specfem::compute::impl::boundary_medium_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::boundary_tag::stacey>;

template class specfem::compute::impl::boundary_medium_container<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    specfem::element::boundary_tag::stacey>;

template class specfem::compute::boundary_value_container<
    specfem::dimension::type::dim2, specfem::element::boundary_tag::stacey>;

specfem::compute::boundary_values::boundary_values(
    const int nstep, const specfem::compute::mesh mesh,
    const specfem::compute::element_types element_types,
    const specfem::compute::boundaries boundaries)
    : stacey(nstep, mesh, element_types, boundaries),
      composite_stacey_dirichlet(nstep, mesh, element_types, boundaries) {}
