#include "specfem/enums.hpp"

#include <gtest/gtest.h>

TEST(IsVectorSeismogram, PressureIsScalarEverywhere) {
  using specfem::element::dimension_tag;
  using specfem::element::medium_tag;
  using specfem::enums::is_vector_seismogram;
  using specfem::enums::wavefield;
  const wavefield type = wavefield::pressure;
  const medium_tag dim2_media[] = {
    medium_tag::elastic_psv, medium_tag::elastic_sh, medium_tag::acoustic,
    medium_tag::poroelastic, medium_tag::elastic_psv_t
  };
  for (const auto medium : dim2_media)
    EXPECT_FALSE(is_vector_seismogram(type, dimension_tag::dim2, medium));

  const medium_tag dim3_media[] = { medium_tag::elastic, medium_tag::acoustic,
                                    medium_tag::elastic_spin };
  for (const auto medium : dim3_media)
    EXPECT_FALSE(is_vector_seismogram(type, dimension_tag::dim3, medium));
}

TEST(IsVectorSeismogram, KinematicTypesAreVectorsExceptSh) {
  using specfem::element::dimension_tag;
  using specfem::element::medium_tag;
  using specfem::enums::is_vector_seismogram;
  using specfem::enums::wavefield;
  const wavefield kinematic[] = { wavefield::displacement, wavefield::velocity,
                                  wavefield::acceleration };
  for (const auto type : kinematic) {
    EXPECT_TRUE(
        is_vector_seismogram(type, dimension_tag::dim3, medium_tag::elastic));
    EXPECT_TRUE(
        is_vector_seismogram(type, dimension_tag::dim3, medium_tag::acoustic));
    EXPECT_TRUE(is_vector_seismogram(type, dimension_tag::dim2,
                                     medium_tag::elastic_psv));
    EXPECT_TRUE(
        is_vector_seismogram(type, dimension_tag::dim2, medium_tag::acoustic));
    EXPECT_TRUE(is_vector_seismogram(type, dimension_tag::dim2,
                                     medium_tag::poroelastic));
    EXPECT_TRUE(is_vector_seismogram(type, dimension_tag::dim2,
                                     medium_tag::elastic_psv_t));
    EXPECT_FALSE(is_vector_seismogram(type, dimension_tag::dim2,
                                      medium_tag::elastic_sh));
  }
}

TEST(IsVectorSeismogram, AxialTypesAreVectorsOnlyIn3d) {
  using specfem::element::dimension_tag;
  using specfem::element::medium_tag;
  using specfem::enums::is_vector_seismogram;
  using specfem::enums::wavefield;
  const wavefield axial[] = { wavefield::rotation,
                              wavefield::intrinsic_rotation, wavefield::curl };
  for (const auto type : axial) {
    EXPECT_TRUE(is_vector_seismogram(type, dimension_tag::dim3,
                                     medium_tag::elastic_spin));
    EXPECT_TRUE(
        is_vector_seismogram(type, dimension_tag::dim3, medium_tag::elastic));
    EXPECT_FALSE(is_vector_seismogram(type, dimension_tag::dim2,
                                      medium_tag::elastic_psv_t));
    EXPECT_FALSE(is_vector_seismogram(type, dimension_tag::dim2,
                                      medium_tag::elastic_psv));
  }
}

TEST(IsVectorSeismogram, IsConstexpr) {
  using specfem::element::dimension_tag;
  using specfem::element::medium_tag;
  using specfem::enums::is_vector_seismogram;
  using specfem::enums::wavefield;
  static_assert(is_vector_seismogram(wavefield::displacement,
                                     dimension_tag::dim3, medium_tag::elastic));
  static_assert(!is_vector_seismogram(wavefield::pressure, dimension_tag::dim3,
                                      medium_tag::acoustic));
  SUCCEED();
}
