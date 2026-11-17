
#include "nonconforming_interfaces.hpp"
#include "impl/interface_container.tpp"
#include "impl/unique_faces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/enums.hpp"
#include "specfem/tag_dispatch/for_each.hpp"

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
    : interface_container(
          [&, hosts = select_hosts(element_intersections, mesh,
                                   flux_scheme_config)]<typename TagsType>() {
            return InterfaceContainerTemplateType<TagsType>(
                ngllz, nglly, ngllx, element_intersections, jacobian_matrix,
                mesh, hosts);
          }){};

specfem::assembly::nonconforming_interfaces_impl::HostSelection
specfem::assembly::nonconforming_interfaces<
    specfem::element::dimension_tag::dim3>::
    select_hosts(
        const specfem::assembly::element_intersections<
            specfem::element::dimension_tag::dim3> &element_intersections,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const specfem::element_coupling::flux_scheme_configuration
            &flux_scheme_config) {
  const auto meshing_type = flux_scheme_config.get_interfacial_meshing_type();
  specfem::assembly::nonconforming_interfaces_impl::HostSelection hosts(
      meshing_type);
  if (meshing_type ==
      specfem::element_coupling::interfacial_meshing_type::self_host) {
    return hosts;
  }
  // Only the self faces are added: a self face belongs to one element with one
  // boundary tag, so the totals of an interface tag need no deduplication.
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    const auto [self_faces, coupled_faces] =
        element_intersections.get_intersections_on_host(
            TagsType::connection_tag, TagsType::interface_tag,
            TagsType::boundary_tag, TagsType::flux_scheme_tag);
    const auto total =
        specfem::assembly::nonconforming_interfaces_impl::face_length_sum(
            self_faces, mesh);
    hosts.add_faces(TagsType::interface_tag, total.length_sum, total.count);
  });
  return hosts;
}
