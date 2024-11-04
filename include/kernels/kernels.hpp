#ifndef _SPECFEM_KERNELS_HPP
#define _SPECFEM_KERNELS_HPP

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "impl/domain_kernels.hpp"

namespace specfem {
namespace kernels {
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType, typename qp_type>
class kernels
    : public impl::domain_kernels<WavefieldType, DimensionType, qp_type> {
public:
  kernels(const type_real dt, const specfem::compute::assembly &assembly,
          const qp_type &quadrature_points)
      : impl::domain_kernels<WavefieldType, DimensionType, qp_type>(
            dt, assembly, quadrature_points) {}
};

} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_HPP */
