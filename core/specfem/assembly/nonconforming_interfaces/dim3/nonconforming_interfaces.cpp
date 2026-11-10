
#include "nonconforming_interfaces.hpp"
#include "impl/interface_container.tpp"
#include "impl/unique_faces.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/element_coupling/to_string.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh_entity/dim3/mesh_entity.hpp"
#include "specfem/tag_dispatch/for_each.hpp"
#include <compare>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace specfem::assembly::nonconforming_interfaces_impl {

/** @brief Identity of one touching face pair, seen from its host side. */
struct PairKey {
  int host_element = -1; ///< element of the host face
  specfem::mesh_entity::dim3::type host_face =
      specfem::mesh_entity::dim3::type::bottom; ///< face type of the host face
  int guest_element = -1;                       ///< element of the guest face
  specfem::mesh_entity::dim3::type guest_face =
      specfem::mesh_entity::dim3::type::bottom; ///< face type of the guest face
  auto operator<=>(const PairKey &) const = default;
};

/**
 * @brief Copy of the three blocks of one row (pair) of a host container.
 *
 * Each block is flattened by `copy_row_from` in the loop order
 * `i` (outer), `j`, then, inside each node, `d` for the normal and `igll`
 * (outer), `axis` (inner) for the interpolants. `face_factor` holds one value
 * per node, `face_normal` the normal components per node and `interpolants`
 * the `igll` x `axis` values per node, all in node order `(i, j)`.
 */
struct HostRow {
  /** @brief `face_factor(iface, i, j)` */
  std::vector<type_real> face_factor{};
  /** @brief `face_normal(iface, i, j, d)` */
  std::vector<type_real> face_normal{};
  /** @brief `interpolants(iface, i, j, igll, axis)` */
  std::vector<type_real> interpolants{};
};

/**
 * @brief Copies row `iface` of the host mirrors of a container.
 * @param container host container
 * @param iface row (pair) index
 * @return copy of the three blocks of the row
 */
template <typename Container>
HostRow copy_row_from(const Container &container, const int iface) {
  const std::size_t ni = container.h_face_factor.extent(1);
  const std::size_t nj = container.h_face_factor.extent(2);
  const std::size_t nd = container.h_face_normal.extent(3);
  const std::size_t ngll = container.h_interpolants.extent(3);
  const std::size_t naxis = container.h_interpolants.extent(4);
  HostRow row;
  for (std::size_t i = 0; i < ni; ++i) {
    for (std::size_t j = 0; j < nj; ++j) {
      row.face_factor.push_back(container.h_face_factor(iface, i, j));
      for (std::size_t d = 0; d < nd; ++d) {
        row.face_normal.push_back(container.h_face_normal(iface, i, j, d));
      }
      for (std::size_t igll = 0; igll < ngll; ++igll) {
        for (std::size_t axis = 0; axis < naxis; ++axis) {
          row.interpolants.push_back(
              container.h_interpolants(iface, i, j, igll, axis));
        }
      }
    }
  }
  return row;
}

/**
 * @brief Writes `row` into row `iface` of the host mirrors of a container, in
 * the order `copy_row_from` fills it.
 * @param container guest container
 * @param iface row (pair) index
 * @param row data to store
 */
template <typename Container>
void store_row_in(Container &container, const int iface, const HostRow &row) {
  const std::size_t ni = container.h_face_factor.extent(1);
  const std::size_t nj = container.h_face_factor.extent(2);
  const std::size_t nd = container.h_face_normal.extent(3);
  const std::size_t ngll = container.h_interpolants.extent(3);
  const std::size_t naxis = container.h_interpolants.extent(4);
  std::size_t n_factor = 0;
  std::size_t n_normal = 0;
  std::size_t n_interp = 0;
  for (std::size_t i = 0; i < ni; ++i) {
    for (std::size_t j = 0; j < nj; ++j) {
      container.h_face_factor(iface, i, j) = row.face_factor[n_factor++];
      for (std::size_t d = 0; d < nd; ++d) {
        container.h_face_normal(iface, i, j, d) = row.face_normal[n_normal++];
      }
      for (std::size_t igll = 0; igll < ngll; ++igll) {
        for (std::size_t axis = 0; axis < naxis; ++axis) {
          container.h_interpolants(iface, i, j, igll, axis) =
              row.interpolants[n_interp++];
        }
      }
    }
  }
}

} // namespace specfem::assembly::nonconforming_interfaces_impl

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
          }) {
  if (flux_scheme_config.get_interfacial_meshing_type() ==
      specfem::element_coupling::interfacial_meshing_type::self_host) {
    // every container is a host: nothing to copy
    return;
  }

  namespace impl = specfem::assembly::nonconforming_interfaces_impl;

  // Host pass: copy the rows of every host container.
  std::map<impl::PairKey, impl::HostRow> host_rows;
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    const auto &container = interface_container.template get<TagsType>();
    if (!container.self_is_host) {
      return;
    }
    const auto [self_faces, coupled_faces] =
        element_intersections.get_intersections_on_host(
            TagsType::connection_tag, TagsType::interface_tag,
            TagsType::boundary_tag, TagsType::flux_scheme_tag);
    for (int iface = 0; iface < static_cast<int>(self_faces.N); ++iface) {
      const auto &self_face = self_faces(iface);
      const auto &coupled_face = coupled_faces(iface);
      const impl::PairKey key{ .host_element = self_face.element_index,
                               .host_face = self_face.face_type,
                               .guest_element = coupled_face.element_index,
                               .guest_face = coupled_face.face_type };
      if (!host_rows.emplace(key, impl::copy_row_from(container, iface))
               .second) {
        std::ostringstream oss;
        oss << "Duplicate nonconforming host pair: interface "
            << specfem::element_coupling::to_string(TagsType::interface_tag)
            << ", boundary "
            << specfem::element::to_string(TagsType::boundary_tag)
            << ", host element " << self_face.element_index << " face "
            << specfem::mesh_entity::dim3::to_string(self_face.face_type)
            << ", guest element " << coupled_face.element_index << " face "
            << specfem::mesh_entity::dim3::to_string(coupled_face.face_type);
        throw std::runtime_error(oss.str());
      }
    }
  });

  // Guest pass: fill every guest container with the rows of its own pairs.
  specfem::tag_dispatch::for_each(combinations, [&]<typename TagsType>() {
    auto &container = interface_container.template get<TagsType>();
    if (container.self_is_host) {
      return;
    }
    const auto [self_faces, coupled_faces] =
        element_intersections.get_intersections_on_host(
            TagsType::connection_tag, TagsType::interface_tag,
            TagsType::boundary_tag, TagsType::flux_scheme_tag);
    for (int iface = 0; iface < static_cast<int>(self_faces.N); ++iface) {
      const auto &self_face = self_faces(iface);
      const auto &coupled_face = coupled_faces(iface);
      const impl::PairKey key{ .host_element = coupled_face.element_index,
                               .host_face = coupled_face.face_type,
                               .guest_element = self_face.element_index,
                               .guest_face = self_face.face_type };
      const auto it = host_rows.find(key);
      if (it == host_rows.end()) {
        std::ostringstream oss;
        oss << "No host row found for nonconforming guest pair: interface "
            << specfem::element_coupling::to_string(TagsType::interface_tag)
            << ", boundary "
            << specfem::element::to_string(TagsType::boundary_tag)
            << ", guest (self) element " << self_face.element_index << " face "
            << specfem::mesh_entity::dim3::to_string(self_face.face_type)
            << ", host (coupled) element " << coupled_face.element_index
            << " face "
            << specfem::mesh_entity::dim3::to_string(coupled_face.face_type);
        throw std::runtime_error(oss.str());
      }
      impl::store_row_in(container, iface, it->second);
    }
    Kokkos::deep_copy(container.face_factor, container.h_face_factor);
    Kokkos::deep_copy(container.face_normal, container.h_face_normal);
    Kokkos::deep_copy(container.interpolants, container.h_interpolants);
  });
}

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
