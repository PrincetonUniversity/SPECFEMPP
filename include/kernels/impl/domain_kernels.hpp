#ifndef _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP
#define _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kernels.hpp"

namespace specfem {
namespace kernels {
namespace impl {
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType, typename qp_type>
class domain_kernels {
public:
  domain_kernels(const type_real dt, const specfem::compute::assembly &assembly,
                 const qp_type &quadrature_points)
      : elastic_kernels(dt, assembly, quadrature_points),
        acoustic_kernels(dt, assembly, quadrature_points) {}

  void prepare_wavefields();

  template <specfem::element::medium_tag medium>
  inline void update_wavefields(const int istep) {
    if constexpr (medium == specfem::element::medium_tag::elastic) {
      elastic_kernels.update_wavefields(istep);
    } else if constexpr (medium == specfem::element::medium_tag::acoustic) {
      acoustic_kernels.update_wavefields(istep);
    }
  }

  void initialize(const specfem::enums::time_scheme::type time_scheme,
                  const type_real &dt) {
    if (time_scheme == specfem::enums::time_scheme::type::newmark) {
      elastic_kernels.initialize_newmark(dt);
      acoustic_kernels.initialize_newmark(dt);
    }

    elastic_kernels.invert_mass_matrix();
    acoustic_kernels.invert_mass_matrix();
    return;
  }

  inline void compute_seismograms(const int &isig_step) {
    elastic_kernels.compute_seismograms(isig_step);
    acoustic_kernels.compute_seismograms(isig_step);
  }

private:
  specfem::kernels::impl::kernels<WavefieldType, DimensionType,
                                  specfem::element::medium_tag::elastic,
                                  qp_type>
      elastic_kernels;
  specfem::kernels::impl::kernels<WavefieldType, DimensionType,
                                  specfem::element::medium_tag::acoustic,
                                  qp_type>
      acoustic_kernels;
};

// template <typename qp_type>
// class domain_kernels<qp_type, specfem::enums::simulation::type::adjoint> {
// public:
//   using elastic_type = specfem::enums::element::medium::elastic;
//   using acoustic_type = specfem::enums::element::medium::acoustic;
//   constexpr static auto forward_type =
//       specfem::enums::simulation::type::forward;
//   constexpr static auto adjoint_type =
//       specfem::enums::simulation::type::adjoint;
//   domain_kernels(const specfem::compute::assembly &assembly,
//                  const qp_type &quadrature_points);

//   void prepare_wavefields();

//   template <specfem::enums::element::medium medium>
//   void update_wavefields(const int istep);

// private:
//   specfem::kernels::impl::domain_kernels_impl<qp_type, forward_type>
//       forward_kernels;
//   specfem::kernels::impl::domain_kernels_impl<qp_type, adjoint_type>
//       adjoint_kernels;
// };
} // namespace impl
} // namespace kernels
} // namespace specfem

#endif /* _SPECFEM_KERNELS_IMPL_DOMAIN_KERNELS_HPP */
