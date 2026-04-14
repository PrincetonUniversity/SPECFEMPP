
#include "impl/interface_container.tpp"
#include "specfem/assembly/conforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"

specfem::assembly::conforming_interfaces<
    specfem::element::dimension_tag::dim2>::
    conforming_interfaces(
        const int ngllz, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim2> &element_intersections,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    interface_container.template get<TagsType>() =
        InterfaceContainerTemplateType<TagsType>(
            ngllz, ngllx, element_intersections, jacobian_matrix, mesh);
  });

  return;
}
