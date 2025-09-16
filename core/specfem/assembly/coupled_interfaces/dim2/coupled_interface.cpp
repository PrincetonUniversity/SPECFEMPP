
#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "impl/interface_container.tpp"
#include "specfem/assembly/coupled_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"

specfem::assembly::coupled_interfaces<specfem::dimension::type::dim2>::
    coupled_interfaces(
        const int ngllz, const int ngllx,
        const specfem::assembly::edge_types<specfem::dimension::type::dim2>
            &edge_types,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(interface_container) {
        _interface_container_ =
            InterfaceContainerType<_interface_tag_, _boundary_tag_>(
                ngllz, ngllx, edge_types, jacobian_matrix, mesh);
      })

  return;
}
