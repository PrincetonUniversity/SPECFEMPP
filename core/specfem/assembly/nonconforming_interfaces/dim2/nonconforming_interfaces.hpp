#pragma once

#include "impl/interface_container.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

template <>
class nonconforming_interfaces<specfem::element::dimension_tag::dim2>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim2> {
public:
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim2;

protected:
  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag>
  using InterfaceContainerType =
      specfem::assembly::nonconforming_interfaces_impl::interface_container<
          dimension_tag, InterfaceTag, BoundaryTag, ConnectionTag>;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE(((InterfaceContainerType,
                                (_INTERFACE_TAG_, _BOUNDARY_TAG_,
                                 _CONNECTION_TAG_)),
                               interface_container)))

public:
  nonconforming_interfaces(
      const int ngllz, const int ngllx,
      const specfem::assembly::edge_types<dimension_tag> &edge_types,
      const specfem::assembly::mesh<dimension_tag> &mesh);

  nonconforming_interfaces() = default;

  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag>
  KOKKOS_INLINE_FUNCTION const
      InterfaceContainerType<InterfaceTag, BoundaryTag, ConnectionTag> &
      get_interface_container() const {
    // Compile-time dispatch using FOR_EACH_IN_PRODUCT macro
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2), CONNECTION_TAG(NONCONFORMING),
                         INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                         BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                      COMPOSITE_STACEY_DIRICHLET)),
                        CAPTURE((interface_container, interface_container)) {
                          if constexpr (InterfaceTag == _interface_tag_ &&
                                        BoundaryTag == _boundary_tag_ &&
                                        ConnectionTag == _connection_tag_) {
                            return _interface_container_;
                          }
                        })

#ifndef NDEBUG
    // Debug check: abort if no matching specialization found
    KOKKOS_ABORT_WITH_LOCATION("specfem::assembly::nonconforming_interfaces::"
                               "get_interface_container(): No "
                               "matching specialization found.");
#endif

    // Unreachable code - satisfy compiler return requirements

    SUPPRESS_TEMPORARY_REF(return {};)
  }
};

} // namespace specfem::assembly

#include "data_access/load.hpp"
