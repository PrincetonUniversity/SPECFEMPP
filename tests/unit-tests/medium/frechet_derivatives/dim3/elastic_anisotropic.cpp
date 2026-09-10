#include "specfem/medium_physics.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>

namespace frechet_derivatives_dim3_anisotropic_impl {

using Tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::anisotropic, false>;

TEST(FrechetDerivatives, ElasticAnisotropic3D) {
  using Acceleration = specfem::point::acceleration<Tags>;
  using Displacement = specfem::point::displacement<Tags>;
  using Derivatives = specfem::point::field_derivatives<Tags>;

  Acceleration adjoint_acceleration;
  Displacement backward_displacement;
  for (int i = 0; i < 3; ++i) {
    adjoint_acceleration(i) = static_cast<type_real>(i + 1);
    backward_displacement(i) = static_cast<type_real>(i + 4);
  }

  Derivatives adjoint_derivatives;
  Derivatives backward_derivatives;
  type_real value = 1.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      adjoint_derivatives.du(i, j) = value;
      backward_derivatives.du(i, j) = value / 2.0;
      value += 1.0;
    }
  }

  const type_real dt = 0.25;
  const auto kernels = specfem::medium_physics::frechet_derivative_impl::
      compute_anisotropic_frechet_derivatives<Tags>(
          adjoint_acceleration, backward_displacement, adjoint_derivatives,
          backward_derivatives, dt);

  const type_real ad[] = { 1.0, 5.0, 9.0, 7.0, 5.0, 3.0 };
  const type_real backward[] = { 0.5, 2.5, 4.5, 3.5, 2.5, 1.5 };
  type_real expected[22] = { -8.0 };
  int index = 1;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      type_real product = ad[i] * backward[j];
      if (j > i) {
        product += ad[j] * backward[i];
        if (j > 2 && i < 3) {
          product *= 2.0;
        }
      }
      if (i > 2) {
        product *= 4.0;
      }
      expected[index++] = -dt * product;
    }
  }

  for (int i = 0; i < 22; ++i) {
    EXPECT_NEAR(kernels[i], expected[i], 1.e-6) << "kernel index " << i;
  }
}

} // namespace frechet_derivatives_dim3_anisotropic_impl
