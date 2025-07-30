#pragma once

#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"
#include "dim2/elastic/isotropic_cosserat/properties_container.hpp"
#include "dim2/poroelastic/isotropic/properties_container.hpp"
#include "enumerations/medium.hpp"
#include "impl/accessor.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
class mesh_to_compute_mapping;
template <specfem::dimension::type Dimension> struct mesh;
} // namespace specfem::assembly

namespace specfem::medium {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct properties_container;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct properties_container<specfem::dimension::type::dim2, MediumTag,
                            PropertyTag>
    : public properties::data_container<specfem::dimension::type::dim2,
                                        MediumTag, PropertyTag>,
      public impl::Accessor<specfem::dimension::type::dim2,
                            properties_container<specfem::dimension::type::dim2,
                                                 MediumTag, PropertyTag> > {

  using base_type = properties::data_container<specfem::dimension::type::dim2,
                                               MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< Dimension of the material
  constexpr static auto medium_tag = base_type::medium_tag; ///< Medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Property type

  properties_container() = default;

  properties_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::assembly::mesh<dimension_tag> &mesh, const int ngllz,
      const int ngllx, const specfem::mesh::materials<dimension_tag> &materials,
      const bool has_gll_model,
      const specfem::kokkos::HostView1d<int> property_index_mapping);

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_device_values(const IndexType &index, PointValues &values) const = delete;

  template <typename PointValues, typename IndexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  add_host_values(const IndexType &index, PointValues &values) const = delete;
};

} // namespace specfem::medium
