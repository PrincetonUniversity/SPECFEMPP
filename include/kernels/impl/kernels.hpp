#ifndef _SPECFEM_KERNELS_IMPL_KERNELS_HPP
#define _SPECFEM_KERNELS_IMPL_KERNELS_HPP

#include "compute/interface.hpp"
#include "domain/domain.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "interface_kernels.hpp"

namespace specfem {
namespace kernels {
namespace impl {
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag, typename qp_type>
class kernels
    : public interface_kernels<WavefieldType, DimensionType, MediumTag> {

public:
  kernels(const type_real dt, const specfem::compute::assembly &assembly,
          const qp_type &quadrature_points)
      : domain(dt, assembly, quadrature_points),
        interface_kernels<WavefieldType, DimensionType, MediumTag>(assembly) {}

  inline void update_wavefields(const int istep) {
    interface_kernels<WavefieldType, DimensionType,
                      MediumTag>::compute_coupling();
    domain.compute_source_interaction(istep);
    domain.compute_stiffness_interaction(istep);
    domain.divide_mass_matrix();
  }

  inline void initialize_newmark(const type_real &dt) {
    domain.template mass_time_contribution<
        specfem::enums::time_scheme::type::newmark>(dt);
  }

  inline void invert_mass_matrix() { domain.invert_mass_matrix(); }

  inline void compute_seismograms(const int &isig_step) {
    domain.compute_seismograms(isig_step);
  }

private:
  specfem::domain::domain<WavefieldType, DimensionType, MediumTag, qp_type>
      domain;
};
} // namespace impl
} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_IMPL_KERNELS_HPP */
