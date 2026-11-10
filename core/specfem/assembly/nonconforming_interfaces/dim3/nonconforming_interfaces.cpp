
#include "nonconforming_interfaces.hpp"
#include "impl/interface_container.tpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/enums.hpp"
#include <sstream>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::nonconforming_interfaces_impl {

/** @brief Host selection of one interface container */
struct HostSelection {
  /// Interface tag of the container
  specfem::element_coupling::interface_tag interface_tag = {};
  /// Conjugate of `interface_tag`
  specfem::element_coupling::interface_tag conjugate_interface_tag = {};
  /// Boundary tag of the container
  specfem::element::boundary_tag boundary_tag = {};
  /// Number of pairs stored
  int num_faces = 0;
  /// Whether the self side is the host
  bool self_is_host = false;
};

/**
 * @brief Throws unless every pair is integrated on exactly one side.
 *
 * The host is selected per container. That selection must agree within one
 * interface tag, and exactly one of an interface tag and its conjugate must
 * be the host (under `self_host` both are). Empty containers are ignored.
 *
 * @param selections Host selection of every container
 * @param interfacial_meshing_type Meshing type used to select the hosts
 */
void check_host_selection(
    const std::vector<HostSelection> &selections,
    const specfem::element_coupling::interfacial_meshing_type
        &interfacial_meshing_type);

} // namespace specfem::assembly::nonconforming_interfaces_impl

void specfem::assembly::nonconforming_interfaces_impl::check_host_selection(
    const std::vector<HostSelection> &selections,
    const specfem::element_coupling::interfacial_meshing_type
        &interfacial_meshing_type) {

  const bool every_side_hosts =
      interfacial_meshing_type ==
      specfem::element_coupling::interfacial_meshing_type::self_host;

  for (const auto &a : selections) {
    for (const auto &b : selections) {
      if (a.num_faces == 0 || b.num_faces == 0)
        continue;
      const bool same_tag_disagree = a.interface_tag == b.interface_tag &&
                                     a.self_is_host != b.self_is_host;
      const bool conjugates =
          !every_side_hosts && b.interface_tag == a.conjugate_interface_tag;
      const bool both_host = conjugates && a.self_is_host && b.self_is_host;
      const bool neither_host =
          conjugates && !a.self_is_host && !b.self_is_host;
      if (!same_tag_disagree && !both_host && !neither_host)
        continue;

      std::ostringstream oss;
      oss << "Inconsistent nonconforming host selection (meshing type "
          << interfacial_meshing_type << "): "
          << (same_tag_disagree
                  ? "containers of the same interface tag disagree"
              : both_host
                  ? "an interface tag and its conjugate are both host"
                  : "neither an interface tag nor its conjugate is host")
          << ".\n";
      for (const auto *s : { &a, &b }) {
        oss << "    interface = " << s->interface_tag
            << ", boundary = " << specfem::element::to_string(s->boundary_tag)
            << ", pairs = " << s->num_faces
            << ", self_is_host = " << (s->self_is_host ? "true" : "false")
            << "\n";
      }
      oss << "Each physical pair is filed under both interface tags and must "
             "be integrated on exactly one side, but the host side is "
             "selected per container.";
      throw std::runtime_error(oss.str());
    }
  }
}

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
