#ifndef ELASTIC_GRADIENT_2D_HPP
#define ELASTIC_GRADIENT_2D_HPP

#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace Domain {
namespace elastic {
namespace impl {
namespace operators {
class gradient2d {

public:
  gradient2d() = default;
  gradient2d(const specfem::compute::partial_derivatives &partial_derivatives)
      : partial_derivatives(partial_derivatives) {}

  template <int NGLL>
  KOKKOS_FUNCTION void operator()(
      const int &xz, const int &ispec,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          s_hprime_xx,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          s_hprime_zz,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          field_x,
      const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
          field_z,
      type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
      type_real &duzdzl) const;

  KOKKOS_FUNCTION void
  operator()(const int &xz, const int &ispec,
             const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx,
             const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz,
             const specfem::kokkos::DeviceScratchView2d<type_real> field_x,
             const specfem::kokkos::DeviceScratchView2d<type_real> field_z,
             type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
             type_real &duzdzl) const;

private:
  specfem::compute::partial_derivatives partial_derivatives;
};
} // namespace operators
} // namespace impl
} // namespace elastic
} // namespace Domain
} // namespace specfem

#endif
