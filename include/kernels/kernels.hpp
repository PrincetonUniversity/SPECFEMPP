#ifndef _SPECFEM_KERNELS_HPP
#define _SPECFEM_KERNELS_HPP

#include "compute/interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "impl/domain_kernels.hpp"

namespace specfem {
namespace kernels {
template <specfem::wavefield::simulation_field WavefieldType,
          specfem::dimension::type DimensionType, int NGLL>
class kernels
    : public impl::domain_kernels<WavefieldType, DimensionType, NGLL> {
public:
  kernels(const type_real dt, const specfem::compute::assembly &assembly)
      : impl::domain_kernels<WavefieldType, DimensionType, NGLL>(dt, assembly) {
  }
};

} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_HPP */
