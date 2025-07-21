#pragma once

#include "dim2/acoustic/isotropic/kernels_container.hpp"
#include "dim2/elastic/anisotropic/kernels_container.hpp"
#include "dim2/elastic/isotropic/kernels_container.hpp"
#include "dim2/elastic/isotropic_cosserat/kernels_container.hpp"
#include "dim2/poroelastic/isotropic/kernels_container.hpp"
#include "impl/accessor.hpp"
#include "impl/data_container.hpp"

namespace specfem {
namespace medium {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct kernels_container;

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct kernels_container<specfem::dimension::type::dim2, MediumTag, PropertyTag>
    : public kernels::data_container<specfem::dimension::type::dim2, MediumTag,
                                     PropertyTag>,
      public impl::Accessor<kernels_container<specfem::dimension::type::dim2,
                                              MediumTag, PropertyTag> > {

  using base_type = kernels::data_container<specfem::dimension::type::dim2,
                                            MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< Dimension of the material
  constexpr static auto medium_tag = base_type::medium_tag; ///< Medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Property type

  kernels_container() = default;

  kernels_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : base_type(elements.extent(0), ngllz, ngllx) {
    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct kernels_container<specfem::dimension::type::dim3, MediumTag, PropertyTag>
    : public kernels::data_container<specfem::dimension::type::dim3, MediumTag,
                                     PropertyTag>,
      public impl::Accessor<kernels_container<specfem::dimension::type::dim3,
                                              MediumTag, PropertyTag> > {

  using base_type = kernels::data_container<specfem::dimension::type::dim3,
                                            MediumTag, PropertyTag>;
  using base_type::base_type;

  constexpr static auto dimension_tag =
      base_type::dimension_tag; ///< Dimension of the material
  constexpr static auto medium_tag = base_type::medium_tag; ///< Medium type
  constexpr static auto property_tag =
      base_type::property_tag; ///< Property type

  kernels_container() = default;

  kernels_container(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int nglly, const int ngllx,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : base_type(elements.extent(0), ngllz, nglly, ngllx) {
    const int nelement = elements.extent(0);
    int count = 0;
    for (int i = 0; i < nelement; ++i) {
      const int ispec = elements(i);
      property_index_mapping(ispec) = count;
      count++;
    }
  }
};
} // namespace medium
} // namespace specfem

// Including the template specializations here so that kernels_container is
// an interface to the compute/kernels module
