
#include "nonconforming_interfaces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"

specfem::assembly::nonconforming_interfaces<
    specfem::element::dimension_tag::dim3>::
    nonconforming_interfaces(
        const int ngllz, const int ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config)
    : interface_container([&]<typename TagsType>() {
        return InterfaceContainerTemplateType<TagsType>(
            ngllz, ngllx, element_intersections, mesh, flux_scheme_config);
      }){};
