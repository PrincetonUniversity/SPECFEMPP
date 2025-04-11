#include "mesh/dim2/coupled_interfaces/interface_container.hpp"
#include "mesh/dim2/coupled_interfaces/interface_container.tpp"
#include "mesh/mesh_base.hpp"
#define GENERATE_INTERFACE_CONTAINER(DimensionType)                            \
  template class specfem::mesh::interface_container<                           \
      DimensionType, specfem::element::medium_tag::elastic_psv,                \
      specfem::element::medium_tag::acoustic>;                                 \
  template class specfem::mesh::interface_container<                           \
      DimensionType, specfem::element::medium_tag::acoustic,                   \
      specfem::element::medium_tag::poroelastic>;                              \
  template class specfem::mesh::interface_container<                           \
      DimensionType, specfem::element::medium_tag::elastic_psv,                \
      specfem::element::medium_tag::poroelastic>;

// Explicitly instantiate template class
GENERATE_INTERFACE_CONTAINER(specfem::dimension::type::dim2)
// GENERATE_INTERFACE_CONTAINER(specfem::dimension::type::dim3)
