#pragma once

#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"
#include "dim2/elastic/isotropic_cosserat/properties_container.hpp"
#include "dim2/poroelastic/isotropic/properties_container.hpp"
#include "enumerations/medium.hpp"
#include "impl/accessor.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace assembly {
class mesh_to_compute_mapping;
}
} // namespace specfem

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct properties_container
    : public properties::data_container<MediumTag, PropertyTag>,
      public impl::Accessor<properties_container<MediumTag, PropertyTag> > {

  using base_type = properties::data_container<MediumTag, PropertyTag>;
  using base_type::base_type;

  properties_container() = default;

  properties_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::assembly::mesh_to_compute_mapping &mapping,
      const int ngllz, const int ngllx,
      const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
      const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping);

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const = delete;

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, PointValues &values) const = delete;
};

} // namespace medium
} // namespace specfem
