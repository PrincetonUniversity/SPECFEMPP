
#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "impl/interface_container.tpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"

specfem::assembly::nonconforming_interfaces<
    specfem::element::dimension_tag::dim2>::
    nonconforming_interfaces(
        const int ngllz, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim2> &element_intersections,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      CAPTURE(interface_container) {
                        _interface_container_ = InterfaceContainerType<
                            _interface_tag_, _boundary_tag_, _connection_tag_>(
                            ngllz, ngllx, element_intersections, mesh);
                      })

  return;
}
