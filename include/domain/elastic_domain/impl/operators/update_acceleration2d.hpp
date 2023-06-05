#ifndef ELASTIC_UPDATE_ACCELERATION_2D_HPP
#define ELASTIC_UPDATE_ACCELERATION_2D_HPP

#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace Domain {
namespace elastic {
namespace impl {
namespace operators {
class update_acceleration2d {

public:
  update_acceleration2d() = default;
  update_acceleration2d(
      specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
          field_dot_dot)
      : field_dot_dot(field_dot_dot){};

  template <int NGLL>
  KOKKOS_FUNCTION void operator()(
      const int &xz, const int &ispec, const int &iglob,
      const type_real &wxglll, const type_real &wzglll,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          stress_integrand_1,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          stress_integrand_2,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          stress_integrand_3,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          stress_integrand_4,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          s_hprimewgll_xx,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          s_hprimewgll_zz) const;

  KOKKOS_FUNCTION void operator()(
      const int &xz, const int &ispec, const int &iglob,
      const type_real &wxglll, const type_real &wzglll,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_1,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_2,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_3,
      const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_4,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx,
      const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz)
      const;

private:
  specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot_dot;
};
} // namespace operators
} // namespace impl
} // namespace elastic
} // namespace Domain
} // namespace specfem

#endif
