#pragma once

#include "impl/interface_container.hpp"
#include "specfem/assembly/element_intersections.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/element_coupling/flux_scheme_configuration.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros/tag_dispatch.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::element::dimension_tag DimensionTag>
struct nonconforming_interfaces;

template <>
class nonconforming_interfaces<specfem::element::dimension_tag::dim3>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::edge,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::element::dimension_tag::dim3> {
public:
  static constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;

protected:
  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag,
            specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
  using InterfaceContainerType =
      specfem::assembly::nonconforming_interfaces_impl::interface_container<
          dimension_tag, InterfaceTag, BoundaryTag, ConnectionTag,
          FluxSchemeTag>;

  template <typename TagsType>
  using InterfaceContainerTemplateType =
      InterfaceContainerType<TagsType::interface_tag, TagsType::boundary_tag,
                             TagsType::connection_tag,
                             TagsType::flux_scheme_tag>;

  static constexpr auto combinations =
      DIMENSION_SET(dim3) * CONNECTION_SET(nonconforming) *
      INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
      BOUNDARY_SET(none, acoustic_free_surface, stacey,
                   composite_stacey_dirichlet) *
      FLUX_SCHEME_SET(natural);

  specfem::tag_dispatch::TypedStorage<InterfaceContainerTemplateType,
                                      decltype(combinations)>
      interface_container;

public:
  nonconforming_interfaces(
      const int ngllz, const int ngllx,
      const specfem::assembly::element_intersections<dimension_tag>
          &element_intersections,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::element_coupling::flux_scheme_configuration
          &flux_scheme_config = {});

  nonconforming_interfaces() = default;

  template <specfem::element_coupling::interface_tag InterfaceTag,
            specfem::element::boundary_tag BoundaryTag,
            specfem::element_connections::type ConnectionTag,
            specfem::element_coupling::flux_scheme_tag FluxSchemeTag>
  KOKKOS_INLINE_FUNCTION const
      InterfaceContainerType<InterfaceTag, BoundaryTag, ConnectionTag,
                             FluxSchemeTag> &
      get_interface_container() const {
    using TagsType = specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                                         ConnectionTag, InterfaceTag,
                                         BoundaryTag, FluxSchemeTag>;
    return interface_container.template get<TagsType>();
  }
};

} // namespace specfem::assembly

#include "data_access/load.hpp"
