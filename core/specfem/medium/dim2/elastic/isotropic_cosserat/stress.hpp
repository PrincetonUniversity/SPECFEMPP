#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_stress_computation_dim2_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup specfem_stress_computation_dim2_elastic_isotropic_cosserat
 * @brief Compute stress tensor for 2D elastic isotropic Cosserat media.
 *
 * Implements constitutive relations for Cosserat (micropolar) elastic media
 * with rotational degrees of freedom. Extends classical elasticity by including
 * couple stresses and asymmetric force stress tensor to capture size effects
 * and microstructural behavior.
 *
 * **Stress components:**
 * - Classical: \f$\sigma_{xx}\f$, \f$\sigma_{zz}\f$ (normal),
 * \f$\sigma_{xz}\f$, \f$\sigma_{zx}\f$ (shear - asymmetric)
 * - Couple stress: \f$\sigma_{c,xy}\f$, \f$\sigma_{c,zy}\f$ (related to
 * rotation gradients)
 *
 * **Material parameters:**
 * - \f$\lambda, \mu\f$: Classical Lamé parameters
 * - \f$\nu\f$: Cosserat coupling parameter (asymmetry)
 * - \f$\mu_c, \nu_c\f$: Couple stress parameters (microstructural length scale)
 *
 * **Constitutive relations:**
 * \f{align}{
 * \sigma_{xx} &= \lambda(\nabla \cdot \mathbf{u}) + 2\mu \frac{\partial
 * u_x}{\partial x} \\
 * \sigma_{zz} &= \lambda(\nabla \cdot \mathbf{u}) + 2\mu \frac{\partial
 * u_z}{\partial z} \\
 * \sigma_{xz} &= \mu\left(\frac{\partial u_z}{\partial x} + \frac{\partial
 * u_x}{\partial z}\right) + \nu\left(\frac{\partial u_z}{\partial x} -
 * \frac{\partial u_x}{\partial z}\right) \\
 * \sigma_{zx} &= \mu\left(\frac{\partial u_x}{\partial z} + \frac{\partial
 * u_z}{\partial x}\right) + \nu\left(\frac{\partial u_x}{\partial z} -
 * \frac{\partial u_z}{\partial x}\right) \\
 * \sigma_{c,xy} &= (\mu_c + \nu_c)\frac{\partial \phi}{\partial x} \\
 * \sigma_{c,zy} &= (\mu_c + \nu_c)\frac{\partial \phi}{\partial z}
 * \f}
 *
 * where \f$\phi\f$ is the microrotation field and \f$\nabla \cdot \mathbf{u} =
 * \frac{\partial u_x}{\partial x} + \frac{\partial u_z}{\partial z}\f$.
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Cosserat material properties (\f$\lambda, \mu, \nu, \mu_c,
 * \nu_c\f$)
 * @param field_derivatives Displacement and rotation gradients
 * @return 3x2 extended stress tensor (force + couple stresses)
 */
template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_psv_t, UseSIMD>
    impl_compute_stress(const specfem::point::properties<
                            specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_psv_t,
                            specfem::element::property_tag::isotropic_cosserat,
                            UseSIMD> &properties,
                        const specfem::point::field_derivatives<
                            specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_psv_t,
                            UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_xz, sigma_zx, sigma_zz, sigma_c_xy, sigma_c_zy;

  sigma_xx = properties.lambda() * (du(0, 0) + du(1, 1)) +
             static_cast<type_real>(2.0) * properties.mu() * du(0, 0);

  sigma_zz = properties.lambda() * (du(0, 0) + du(1, 1)) +
             static_cast<type_real>(2.0) * properties.mu() * du(1, 1);

  // From Jeroen's spin notes:
  // note that du(0, 1) is the $\partial u_x / \partial z$
  // and du(1, 0) is the $\partial u_z / \partial x$, etc.
  sigma_xz = properties.mu() * (du(1, 0) + du(0, 1)) +
             properties.nu() * (du(1, 0) - du(0, 1));

  sigma_zx = properties.mu() * (du(0, 1) + du(1, 0)) +
             properties.nu() * (du(0, 1) - du(1, 0));

  // Couple stress components for psv propagation
  sigma_c_xy = (properties.mu_c() + properties.nu_c()) * du(2, 0);

  sigma_c_zy = (properties.mu_c() + properties.nu_c()) * du(2, 1);

  specfem::datatype::TensorPointViewType<type_real, 3, 2, UseSIMD> T;

  // Note that the the spin notes have the divergence act on the first component
  // Komatitsch & Tromp (1999) which we are following here defines the
  // divergence as acting on the second component. so we have to implement the
  // transpose
  T(0, 0) = sigma_xx;
  T(1, 0) = sigma_xz;
  T(0, 1) = sigma_zx;
  T(1, 1) = sigma_zz;
  T(2, 0) = sigma_c_xy;
  T(2, 1) = sigma_c_zy;

  return { T };
}

} // namespace medium
} // namespace specfem
