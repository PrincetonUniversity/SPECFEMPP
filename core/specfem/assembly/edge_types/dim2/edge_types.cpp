#include "specfem/assembly/edge_types.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/material_definitions.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/element_types.hpp"
#include <Kokkos_Core.hpp>

using EdgeViewType =
    Kokkos::View<specfem::mesh_entity::edge *, Kokkos::DefaultExecutionSpace>;

specfem::assembly::edge_types<specfem::dimension::type::dim2>::edge_types(
    const int ngllx, const int ngllz,
    const specfem::assembly::mesh<dimension> &mesh,
    const specfem::assembly::element_types<dimension> &element_types,
    const specfem::mesh::coupled_interfaces<dimension> &coupled_interfaces) {

  // Count the number of interfaces for each combination of connection
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_self_edges, h_coupled_edges, self_edges, coupled_edges) {
        int count = 0;
        constexpr auto self_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::self_medium();
        constexpr auto coupled_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::coupled_medium();
        if (_connection_tag_ == specfem::connections::type::weakly_conforming) {
          const auto interface_container =
              coupled_interfaces.template get<self_medium, coupled_medium>();
          const int nedges =
              interface_container.num_interfaces; // number of edges
          for (int iedge = 0; iedge < nedges; ++iedge) {
            const int ispec1_mesh =
                interface_container.medium1_index_mapping(iedge);
            const int ispec2_mesh =
                interface_container.medium2_index_mapping(iedge);
            const int ispec1 = mesh.mesh_to_compute(ispec1_mesh);
            const auto boundary_tag = element_types.get_boundary_tag(ispec1);
            if (boundary_tag == _boundary_tag_) {
              count++;
            }
          }

          _self_edges_ = EdgeViewType(
              "specfem::assembly::interface_types::self_edges", count);
          _coupled_edges_ = EdgeViewType(
              "specfem::assembly::interface_types::coupled_edges", count);
          _h_self_edges_ = Kokkos::create_mirror_view(_self_edges_);
          _h_coupled_edges_ = Kokkos::create_mirror_view(_coupled_edges_);
        }
      })

  static const auto connection_mapping =
      specfem::connections::connection_mapping(ngllx, ngllz);

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_self_edges, h_coupled_edges, self_edges, coupled_edges) {
        int index = 0;
        constexpr auto self_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::self_medium();
        constexpr auto coupled_medium =
            specfem::interface::attributes<_dimension_tag_,
                                           _interface_tag_>::coupled_medium();
        if (_connection_tag_ == specfem::connections::type::weakly_conforming) {
          const auto interface_container =
              coupled_interfaces.template get<self_medium, coupled_medium>();
          const int nedges =
              interface_container.num_interfaces; // number of edges
          for (int iedge = 0; iedge < nedges; ++iedge) {
            const int ispec1_mesh =
                interface_container.medium1_index_mapping(iedge);
            const int ispec2_mesh =
                interface_container.medium2_index_mapping(iedge);
            const int ispec1 = mesh.mesh_to_compute(ispec1_mesh);
            const int ispec2 = mesh.mesh_to_compute(ispec2_mesh);
            const auto boundary_tag = element_types.get_boundary_tag(ispec1);
            if (boundary_tag == _boundary_tag_) {
              const auto edge1 = interface_container.medium1_edge_type(iedge);
              const auto edge2 = interface_container.medium2_edge_type(iedge);
              const auto flip =
                  connection_mapping.flip_orientation(edge1, edge2);
              _h_self_edges_(index) =
                  specfem::mesh_entity::edge{ ispec1, edge1, false };
              _h_coupled_edges_(index) =
                  specfem::mesh_entity::edge{ ispec2, edge2, flip };
              index++;
            }
          }
          Kokkos::deep_copy(_self_edges_, _h_self_edges_);
          Kokkos::deep_copy(_coupled_edges_, _h_coupled_edges_);
        }
      })

  return;
}

std::tuple<EdgeViewType::HostMirror, EdgeViewType::HostMirror>
specfem::assembly::edge_types<specfem::dimension::type::dim2>::
    get_edges_on_host(const specfem::connections::type connection,
                      const specfem::interface::interface_tag edge,
                      const specfem::element::boundary_tag boundary) const {

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                    COMPOSITE_STACEY_DIRICHLET)),
      CAPTURE(h_self_edges, h_coupled_edges) {
        if (_connection_tag_ == connection && _interface_tag_ == edge &&
            _boundary_tag_ == boundary) {
          return std::make_tuple(_h_self_edges_, _h_coupled_edges_);
        }
      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}

std::tuple<EdgeViewType, EdgeViewType>
specfem::assembly::edge_types<specfem::dimension::type::dim2>::
    get_edges_on_device(const specfem::connections::type connection,
                        const specfem::interface::interface_tag edge,
                        const specfem::element::boundary_tag boundary) const {

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(WEAKLY_CONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, STACEY, ACOUSTIC_FREE_SURFACE,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      CAPTURE(self_edges, coupled_edges) {
                        if (_connection_tag_ == connection &&
                            _interface_tag_ == edge &&
                            _boundary_tag_ == boundary) {
                          return std::make_tuple(_self_edges_, _coupled_edges_);
                        }
                      })

  throw std::runtime_error(
      "Connection type, interface type or boundary type not found");
}
