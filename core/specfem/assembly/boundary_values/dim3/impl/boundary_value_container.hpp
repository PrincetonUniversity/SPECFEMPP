#pragma once

#include "boundary_medium_container.hpp"
#include "enumerations/interface.hpp"
#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/macros.hpp"

namespace specfem::assembly::boundary_values_impl {

template <specfem::element::boundary_tag BoundaryTag>
class boundary_value_container<specfem::dimension::type::dim3, BoundaryTag> {

private:
  template <specfem::dimension::type _DimensionTag,
            specfem::element::medium_tag _MediumTag>
  using _boundary_medium_container =
      boundary_medium_container<_DimensionTag, _MediumTag, BoundaryTag>;

  using IndexViewType = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;
  constexpr static auto boundary_tag = BoundaryTag;

  IndexViewType property_index_mapping;
  IndexViewType::HostMirror h_property_index_mapping;

  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
                      DECLARE(((_boundary_medium_container,
                                (_DIMENSION_TAG_, _MEDIUM_TAG_)),
                               container)))

  boundary_value_container() = default;

  boundary_value_container(
      const int nstep, const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);

  void sync_to_host() {
    Kokkos::deep_copy(h_property_index_mapping, property_index_mapping);
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
                        CAPTURE(container) { _container_.sync_to_host(); });
  }

  void sync_to_device() {
    Kokkos::deep_copy(property_index_mapping, h_property_index_mapping);
    FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC, ACOUSTIC)),
                        CAPTURE(container) { _container_.sync_to_device(); });
  }
};
} // namespace specfem::assembly::boundary_values_impl
