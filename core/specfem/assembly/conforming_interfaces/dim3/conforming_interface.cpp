
#include "impl/interface_container.tpp"
#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"

specfem::assembly::conforming_interfaces<
    specfem::element::dimension_tag::dim3>::
    conforming_interfaces(
        const int ngllz, const int nglly, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM3), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(interface_container) {
        _interface_container_ =
            InterfaceContainerType<_interface_tag_, _boundary_tag_,
                                   _connection_tag_>(ngllz, nglly, ngllx,
                                                     element_intersections,
                                                     jacobian_matrix, mesh);
      })

  return;
}
