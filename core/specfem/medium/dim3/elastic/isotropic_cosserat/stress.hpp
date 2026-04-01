#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_stress_computation_dim3_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup specfem_stress_computation_dim3_elastic_isotropic_cosserat
 * @brief Compute stress tensor for 3D elastic isotropic Cosserat media.
 *
 * Implements constitutive relations for Cosserat (micropolar) elastic media
 * with rotational degrees of freedom. Extends classical elasticity by including
 * couple stresses and asymmetric force stress tensor to capture size effects
 * and microstructural behavior.
 *
 * **Stress components:**
 * - Classical: \f$\sigma_{xx}\f$, \f$\sigma_{yy}\f$,\f$\sigma_{zz}\f$ (normal),
 * \f$\sigma_{xy}\f$, \f$\sigma_{yx}\f$, \f$\sigma_{xz}\f$, \f$\sigma_{zx}\f$,
 * \f$\sigma_{yz}\f$, \f$\sigma_{zy}\f$ (shear - asymmetric)
 * - Couple stress: \f$\sigma_{c,xx}\f$, \f$\sigma_{c,yy}\f$,
 * \f$\sigma_{c,zz}\f$,
 * \f$\sigma_{c,xy}\f$, \f$\sigma_{c,yx}\f$, \f$\sigma_{c,xz}\f$,
 * \f$\sigma_{c,zx}\f$, \f$\sigma_{c,yz}\f$, \f$\sigma_{c,zy}\f$ (related to
 * rotation gradients)
 *
 * **Material parameters:**
 * - \f$\lambda, \mu\f$: Classical Lamé parameters
 * - \f$\nu\f$: Cosserat coupling parameter (asymmetry)
 * - \f$\lambda_c, \mu_c, \nu_c\f$: Couple stress parameters (microstructural
 * length scale)
 *
 * **Constitutive relations:**
 * \f{align}{
 * \sigma_{xx} &= \lambda(\nabla \cdot \mathbf{u}) + 2\mu \frac{\partial
 * u_x}{\partial x} \\
 * \sigma_{yy} &= \lambda(\nabla \cdot \mathbf{u}) + 2\mu \frac{\partial
 * u_y}{\partial y} \\
 * \sigma_{zz} &= \lambda(\nabla \cdot \mathbf{u}) + 2\mu \frac{\partial
 * u_z}{\partial z} \\
 * \sigma_{xy} &= \mu\left(\frac{\partial u_y}{\partial x} + \frac{\partial
 * u_x}{\partial y}\right) + \nu\left(\frac{\partial u_y}{\partial x} -
 * \frac{\partial u_x}{\partial y}\right) \\
 * \sigma_{yx} &= \mu\left(\frac{\partial u_x}{\partial y} + \frac{\partial
 * u_y}{\partial x}\right) + \nu\left(\frac{\partial u_x}{\partial y} -
 * \frac{\partial u_y}{\partial x}\right) \\
 * \sigma_{xz} &= \mu\left(\frac{\partial u_z}{\partial x} + \frac{\partial
 * u_x}{\partial z}\right) + \nu\left(\frac{\partial u_z}{\partial x} -
 * \frac{\partial u_x}{\partial z}\right) \\
 * \sigma_{zx} &= \mu\left(\frac{\partial u_x}{\partial z} + \frac{\partial
 * u_z}{\partial x}\right) + \nu\left(\frac{\partial u_x}{\partial z} -
 * \frac{\partial u_z}{\partial x}\right) \\
 * \sigma_{yz} &= \mu\left(\frac{\partial u_z}{\partial y} + \frac{\partial
 * u_y}{\partial z}\right) + \nu\left(\frac{\partial u_z}{\partial y} -
 * \frac{\partial u_y}{\partial z}\right) \\
 * \sigma_{zy} &= \mu\left(\frac{\partial u_y}{\partial z} + \frac{\partial
 * u_z}{\partial y}\right) + \nu\left(\frac{\partial u_y}{\partial z} -
 * \frac{\partial u_z}{\partial y}\right) \\
 * \sigma_{c,xx} &= \lambda_c(\nabla\cdot \mathbf{\phi}) + 2\mu_c \frac{\partial
 * \phi_x}{\partial x} \\
 * \sigma_{c,yy} &= \lambda_c(\nabla\cdot \mathbf{\phi}) + 2\mu_c \frac{\partial
 * \phi_y}{\partial y} \\
 * \sigma_{c,zz} &= \lambda_c(\nabla\cdot \mathbf{\phi}) + 2\mu_c \frac{\partial
 * \phi_z}{\partial z} \\
 * \sigma_{c,xy} &= \mu_c\left(\frac{\partial \phi_y}{\partial x} +
 * \frac{\partial
 * \phi_x}{\partial y}\right) + \nu_c\left(\frac{\partial \phi_y}{\partial x} -
 * \frac{\partial \phi_x}{\partial y}\right) \\
 * \sigma_{c,yx} &= \mu_c\left(\frac{\partial \phi_x}{\partial y} +
 * \frac{\partial
 * \phi_y}{\partial x}\right) + \nu_c\left(\frac{\partial \phi_x}{\partial y} -
 * \frac{\partial \phi_y}{\partial x}\right) \\
 * \sigma_{c,xz} &= \mu_c\left(\frac{\partial \phi_z}{\partial x} +
 * \frac{\partial
 * \phi_x}{\partial z}\right) + \nu_c\left(\frac{\partial \phi_z}{\partial x} -
 * \frac{\partial \phi_x}{\partial z}\right) \\
 * \sigma_{c,zx} &= \mu_c\left(\frac{\partial \phi_x}{\partial z} +
 * \frac{\partial
 * \phi_z}{\partial x}\right) + \nu_c\left(\frac{\partial \phi_x}{\partial z} -
 * \frac{\partial \phi_z}{\partial x}\right) \\
 * \sigma_{c,yz} &= \mu_c\left(\frac{\partial \phi_z}{\partial y} +
 * \frac{\partial
 * \phi_y}{\partial z}\right) + \nu_c\left(\frac{\partial \phi_z}{\partial y} -
 * \frac{\partial \phi_y}{\partial z}\right) \\
 * \sigma_{c,zy} &= \mu_c\left(\frac{\partial \phi_y}{\partial z} +
 * \frac{\partial
 * \phi_z}{\partial y}\right) + \nu_c\left(\frac{\partial \phi_y}{\partial z} -
 * \frac{\partial \phi_z}{\partial y}\right)
 * \f}
 *
 * where \f$\phi\f$ is the microrotation field and \f$\nabla \cdot \mathbf{u} =
 * \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} +
 * \frac{\partial u_z}{\partial z}\f$.
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance
 * @param properties Cosserat material properties (\f$\lambda, \mu, \nu,
 * \lambda_c,
 * \mu_c, \nu_c\f$)
 * @param field_derivatives Displacement and rotation gradients
 * @return 6x3 extended stress tensor (force + couple stresses)
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic_spin &&
            Tags::property_tag ==
                specfem::element::property_tag::isotropic_cosserat,
        int> = 0>
KOKKOS_INLINE_FUNCTION specfem::point::stress<Tags> impl_compute_stress(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::field_derivatives<Tags> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yx, sigma_xz, sigma_zx,
      sigma_yz, sigma_zy;
  datatype sigma_c_xx, sigma_c_yy, sigma_c_zz, sigma_c_xy, sigma_c_yx,
      sigma_c_xz, sigma_c_zx, sigma_c_yz, sigma_c_zy;

  // From Jeroen's spin notes:
  // note that du(i, j) is the $\partial u_i / \partial x_j$
  // when i=0,1,2 (corresponding to x,y,z) and when i=3,4,5,
  // it is the $\partial \phi_i / \partial x_j$ instead.
  sigma_xx = properties.lambda() * (du(0, 0) + du(1, 1) + du(2, 2)) +
             static_cast<type_real>(2.0) * properties.mu() * du(0, 0);

  sigma_yy = properties.lambda() * (du(0, 0) + du(1, 1) + du(2, 2)) +
             static_cast<type_real>(2.0) * properties.mu() * du(1, 1);

  sigma_zz = properties.lambda() * (du(0, 0) + du(1, 1) + du(2, 2)) +
             static_cast<type_real>(2.0) * properties.mu() * du(2, 2);

  sigma_xy = properties.mu() * (du(1, 0) + du(0, 1)) +
             properties.nu() * (du(1, 0) - du(0, 1));

  sigma_yx = properties.mu() * (du(0, 1) + du(1, 0)) +
             properties.nu() * (du(0, 1) - du(1, 0));

  sigma_xz = properties.mu() * (du(2, 0) + du(0, 2)) +
             properties.nu() * (du(2, 0) - du(0, 2));

  sigma_zx = properties.mu() * (du(0, 2) + du(2, 0)) +
             properties.nu() * (du(0, 2) - du(2, 0));

  sigma_yz = properties.mu() * (du(2, 1) + du(1, 2)) +
             properties.nu() * (du(2, 1) - du(1, 2));

  sigma_zy = properties.mu() * (du(1, 2) + du(2, 1)) +
             properties.nu() * (du(1, 2) - du(2, 1));

  // Couple stress components
  sigma_c_xx = properties.lambda_c() * (du(3, 0) + du(4, 1) + du(5, 2)) +
               static_cast<type_real>(2.0) * properties.mu_c() * du(3, 0);

  sigma_c_yy = properties.lambda_c() * (du(3, 0) + du(4, 1) + du(5, 2)) +
               static_cast<type_real>(2.0) * properties.mu_c() * du(4, 1);

  sigma_c_zz = properties.lambda_c() * (du(3, 0) + du(4, 1) + du(5, 2)) +
               static_cast<type_real>(2.0) * properties.mu_c() * du(5, 2);

  sigma_c_xy = properties.mu_c() * (du(4, 0) + du(3, 1)) +
               properties.nu_c() * (du(4, 0) - du(3, 1));

  sigma_c_yx = properties.mu_c() * (du(3, 1) + du(4, 0)) +
               properties.nu_c() * (du(3, 1) - du(4, 0));

  sigma_c_xz = properties.mu_c() * (du(5, 0) + du(3, 2)) +
               properties.nu_c() * (du(5, 0) - du(3, 2));

  sigma_c_zx = properties.mu_c() * (du(3, 2) + du(5, 0)) +
               properties.nu_c() * (du(3, 2) - du(5, 0));

  sigma_c_yz = properties.mu_c() * (du(5, 1) + du(4, 2)) +
               properties.nu_c() * (du(5, 1) - du(4, 2));

  sigma_c_zy = properties.mu_c() * (du(4, 2) + du(5, 1)) +
               properties.nu_c() * (du(4, 2) - du(5, 1));

  specfem::datatype::TensorPointViewType<type_real, 6, 3, Tags::using_simd> T;

  // Note that the the spin notes have the divergence act on the first component
  // Komatitsch & Tromp (1999) which we are following here defines the
  // divergence as acting on the second component. so we have to implement the
  // transpose
  T(0, 0) = sigma_xx;
  T(1, 0) = sigma_xy;
  T(2, 0) = sigma_xz;
  T(0, 1) = sigma_yx;
  T(1, 1) = sigma_yy;
  T(2, 1) = sigma_yz;
  T(0, 2) = sigma_zx;
  T(1, 2) = sigma_zy;
  T(2, 2) = sigma_zz;

  T(3, 0) = sigma_c_xx;
  T(4, 0) = sigma_c_xy;
  T(5, 0) = sigma_c_xz;
  T(3, 1) = sigma_c_yx;
  T(4, 1) = sigma_c_yy;
  T(5, 1) = sigma_c_yz;
  T(3, 2) = sigma_c_zx;
  T(4, 2) = sigma_c_zy;
  T(5, 2) = sigma_c_zz;

  return { T };
}

} // namespace medium_physics
} // namespace specfem
