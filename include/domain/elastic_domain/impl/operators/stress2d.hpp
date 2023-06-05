#ifndef ELASTIC_STRESS_2D_HPP
#define ELASTIC_STRESS_2D_HPP

#include "compute/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"

namespace specfem {
namespace Domain {
namespace elastic {
namespace impl {
namespace operators {
class stress2d {

public:
  stress2d() = default;
  stress2d(const specfem::compute::partial_derivatives &partial_derivatives,
           const specfem ::compute::properties &properties)
      : partial_derivatives(partial_derivatives), properties(properties) {}

  template <int NGLL>
  KOKKOS_FUNCTION void
  operator()(const int &xz, const int &ispec, const type_real &duxdxl,
             const type_real &duxdzl, const type_real &duzdxl,
             const type_real &duzdzl, type_real &stemp_1, type_real &stemp_2,
             type_real &stemp_3, type_real &s_temp4) const;

  KOKKOS_FUNCTION void
  operator()(const int &xz, const int &ispec, const int &ngllx,
             const type_real &duxdxl, const type_real &duxdzl,
             const type_real &duzdxl, const type_real &duzdzl,
             type_real &stress_integrand_1l, type_real &stress_integrand_2l,
             type_real &stress_integrand_3l,
             type_real &stress_integrand_4l) const;

private:
  specfem::compute::partial_derivatives partial_derivatives;
  specfem::compute::properties properties;
};
} // namespace operators
} // namespace impl
} // namespace elastic
} // namespace Domain
} // namespace specfem

#endif
