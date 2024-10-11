#include "mesh/coupled_interfaces/interface_container.hpp"
#include "mesh/coupled_interfaces/interface_container.tpp"

// Explicitly instantiate template class

template class specfem::mesh::interface_container<
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::mesh::interface_container<
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::poroelastic>;

template class specfem::mesh::interface_container<
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::poroelastic>;
