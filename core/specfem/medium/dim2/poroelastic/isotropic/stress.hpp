#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_stress_computation_dim2_poroelastic
 *
 */

/**
 * @ingroup specfem_stress_computation_dim2_poroelastic
 * @brief Compute stress tensor for 2D poroelastic isotropic media.
 *
 * Implements Biot's constitutive relations for fluid-saturated porous media
 * with solid-fluid coupling. Computes both solid stress tensor and fluid
 * pressure from solid and fluid displacement gradients.
 *
 * **Stress components:**
 * - Solid stress: \f$\sigma_{xx}\f$, \f$\sigma_{zz}\f$, \f$\sigma_{xz}\f$
 * (modified by fluid pressure coupling)
 * - Fluid pressure: \f$\sigma_p\f$ (coupled to solid deformation)
 *
 * **Key physics:**
 * - Biot coupling: \f$C_{Biot}\f$ links solid deformation to fluid pressure
 * - Fluid modulus: \f$M_{Biot}\f$ governs fluid compressibility
 * - Porosity effects: \f$\phi/\alpha\f$ coupling between fluid and solid phases
 *
 * **Constitutive relations:**
 * \f{eqnarray}{
 * \sigma_{xx} &=& (\lambda + 2\mu)_G \frac{\partial u_x}{\partial x} +
 * \lambda_G \frac{\partial u_z}{\partial z} + C_{Biot}\left(\frac{\partial
 * w_x}{\partial x} + \frac{\partial w_z}{\partial z}\right) \\
 * \sigma_{zz} &=& (\lambda + 2\mu)_G \frac{\partial u_z}{\partial z} +
 * \lambda_G \frac{\partial u_x}{\partial x} + C_{Biot}\left(\frac{\partial
 * w_x}{\partial x} + \frac{\partial w_z}{\partial z}\right) \\
 * \sigma_{xz} &=& \mu_G \left(\frac{\partial u_z}{\partial x} + \frac{\partial
 * u_x}{\partial z}\right) \\
 * \sigma_p &=& C_{Biot}\left(\frac{\partial u_x}{\partial x} + \frac{\partial
 * u_z}{\partial z}\right) + M_{Biot}\left(\frac{\partial w_x}{\partial x} +
 * \frac{\partial w_z}{\partial z}\right)
 * \f}
 *
 * **Variables:**
 * - \f$u_x, u_z\f$: Solid displacements
 * - \f$w_x, w_z\f$: Fluid displacements relative to solid
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Poroelastic material properties (Biot parameters)
 * @param field_derivatives Solid and fluid displacement gradients
 * @return 4x2 coupled stress tensor (solid + fluid)
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::poroelastic, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::poroelastic,
            specfem::element::property_tag::isotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::poroelastic, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz, sigmap;

  // Poroelastic case
  //  sigma_xx = lambdalplus2mul_G*dux_dxl + lambdal_G*duz_dzl + C_biot*(dwx_dxl
  //  + dwz_dzl) sigma_xz = mu_G*(duz_dxl + dux_dzl) sigma_zz =
  //  lambdalplus2mul_G*duz_dzl + lambdal_G*dux_dxl + C_biot*(dwx_dxl + dwz_dzl)
  //  sigmap = C_biot*(dux_dxl + duz_dzl) + M_biot*(dwx_dxl + dwz_dzl)

  // sigma_xx
  sigma_xx = properties.lambdaplus2mu_G() * du(0, 0) +
             properties.lambda_G() * du(1, 1) +
             properties.C_Biot() * (du(2, 0) + du(3, 1));

  // sigma_zz
  sigma_zz = properties.lambdaplus2mu_G() * du(1, 1) +
             properties.lambda_G() * du(0, 0) +
             properties.C_Biot() * (du(2, 0) + du(3, 1));

  // sigma_xz
  sigma_xz = properties.mu_G() * (du(0, 1) + du(1, 0));

  // sigmap
  sigmap = properties.C_Biot() * (du(0, 0) + du(1, 1)) +
           properties.M_Biot() * (du(2, 0) + du(3, 1));

  specfem::datatype::TensorPointViewType<type_real, 4, 2, UseSIMD> T;

  T(0, 0) = sigma_xx - properties.phi() / properties.tortuosity() * sigmap;
  T(1, 0) = sigma_xz;
  T(0, 1) = sigma_xz;
  T(1, 1) = sigma_zz - properties.phi() / properties.tortuosity() * sigmap;
  T(2, 0) = sigmap - properties.rho_f() / properties.rho_bar() * sigma_xx;
  T(3, 0) = static_cast<type_real>(-1.0) * properties.rho_f() /
            properties.rho_bar() * sigma_xz;
  T(2, 1) = static_cast<type_real>(-1.0) * properties.rho_f() /
            properties.rho_bar() * sigma_xz;
  T(3, 1) = sigmap - properties.rho_f() / properties.rho_bar() * sigma_zz;

  return { T };
}

} // namespace medium
} // namespace specfem
