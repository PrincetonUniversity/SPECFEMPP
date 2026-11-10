#include "host_selection.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/element_coupling/to_string.hpp"
#include <sstream>
#include <stdexcept>

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
