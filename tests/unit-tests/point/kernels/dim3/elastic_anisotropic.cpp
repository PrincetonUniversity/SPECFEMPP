#include "specfem/point/kernels.hpp"
#include "specfem/utilities.hpp"
#include <gtest/gtest.h>

TEST(PointKernels, ElasticAnisotropic3D) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::anisotropic, false>;
  using Kernels = specfem::point::kernels<Tags>;

  const Kernels kernels(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0);

  static_assert(Kernels::nprops == 22);
  for (int i = 0; i < Kernels::nprops; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(kernels[i],
                                             static_cast<type_real>(i + 1)));
  }

  EXPECT_EQ(kernels.rho(), 1.0);
  EXPECT_EQ(kernels.c11(), 2.0);
  EXPECT_EQ(kernels.c16(), 7.0);
  EXPECT_EQ(kernels.c22(), 8.0);
  EXPECT_EQ(kernels.c33(), 13.0);
  EXPECT_EQ(kernels.c44(), 17.0);
  EXPECT_EQ(kernels.c55(), 20.0);
  EXPECT_EQ(kernels.c66(), 22.0);
}
