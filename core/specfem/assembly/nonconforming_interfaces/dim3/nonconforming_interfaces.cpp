
#include "nonconforming_interfaces.hpp"
#include "impl/host_selection.hpp"
#include "impl/interface_container.tpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/enums.hpp"
#include <sstream>
#include <stdexcept>
#include <vector>

specfem::assembly::nonconforming_interfaces<
    specfem::element::dimension_tag::dim3>::
    nonconforming_interfaces(
        const int &ngllz, const int &nglly, const int &ngllx,
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config)
    : interface_container([&]<typename TagsType>() {
        return InterfaceContainerTemplateType<TagsType>(
            ngllz, nglly, ngllx, element_intersections, jacobian_matrix, mesh,
            flux_scheme_config);
      }) {

  std::vector<nonconforming_interfaces_impl::HostSelection> selections;
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    const auto &container = interface_container.template get<TagsType>();
    selections.push_back(
        { TagsType::interface_tag,
          specfem::element_coupling::attributes<
              dimension_tag, TagsType::interface_tag>::conjugate_interface(),
          TagsType::boundary_tag,
          static_cast<int>(container.face_factor.extent(0)),
          container.self_is_host });
  });
  nonconforming_interfaces_impl::check_host_selection(
      selections, flux_scheme_config.get_interfacial_meshing_type());
}
