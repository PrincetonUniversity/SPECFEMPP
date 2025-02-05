#pragma once

#include "compute/properties/properties.hpp"
#include "enumerations/interface.hpp"
#include "kernels_container.hpp"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
class material_kernels : public kernels_container<type, property> {
public:
  constexpr static auto value_type = type;
  constexpr static auto property_type = property;

  material_kernels() = default;

  material_kernels(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const int ngllz, const int ngllx,
      const specfem::kokkos::HostView1d<int> property_index_mapping)
      : specfem::medium::kernels_container<value_type, property_type>(
            elements.extent(0), ngllz, ngllx) {

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
