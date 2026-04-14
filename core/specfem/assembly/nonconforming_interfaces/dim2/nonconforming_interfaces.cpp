
#include "specfem/assembly/nonconforming_interfaces.hpp"
#include "impl/interface_container.tpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"

specfem::assembly::nonconforming_interfaces<
    specfem::element::dimension_tag::dim2>::
    nonconforming_interfaces(
        const int ngllz, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim2> &element_intersections,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    interface_container.template get<TagsType>() =
        InterfaceContainerTemplateType<TagsType>(ngllz, ngllx,
                                                 element_intersections, mesh);
  });

  return;
}
