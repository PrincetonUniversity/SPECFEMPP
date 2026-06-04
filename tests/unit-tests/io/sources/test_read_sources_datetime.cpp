#include "specfem/datetime.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/sources/impl/reader.hpp"
#include "specfem/source.hpp"
#include <gtest/gtest.h>

// Shared constants (defined in test_source_solutions.cpp)
extern int nsteps;
extern type_real dt;

// ---------------------------------------------------------------------------
// 2D tests
// ---------------------------------------------------------------------------

TEST(ReadSourcesDatetime2D, GlobalDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim2/sources_global_datetime.yaml"),
          nsteps, dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 2u);

  auto expected = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);

  ASSERT_TRUE(sources[0]->get_starttime().has_value());
  ASSERT_TRUE(sources[1]->get_starttime().has_value());
  EXPECT_EQ(*sources[0]->get_starttime(), expected);
  EXPECT_EQ(*sources[1]->get_starttime(), expected);
}

TEST(ReadSourcesDatetime2D, PerSourceDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim2/sources_per_source_datetime.yaml"),
          nsteps, dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 2u);

  auto expected_0 = specfem::datetime::make(2003, 12, 26, 1, 56, 50.0);
  auto expected_1 = specfem::datetime::make(2003, 12, 26, 1, 56, 52.0);

  ASSERT_TRUE(sources[0]->get_starttime().has_value());
  ASSERT_TRUE(sources[1]->get_starttime().has_value());
  EXPECT_EQ(*sources[0]->get_starttime(), expected_0);
  EXPECT_EQ(*sources[1]->get_starttime(), expected_1);
}

TEST(ReadSourcesDatetime2D, NoDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim2,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim2/single_force.yaml"), nsteps, dt,
          specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 1u);
  EXPECT_FALSE(sources[0]->get_starttime().has_value());
}

// ---------------------------------------------------------------------------
// 3D tests
// ---------------------------------------------------------------------------

TEST(ReadSourcesDatetime3D, GlobalDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim3,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim3/sources_global_datetime.yaml"),
          nsteps, dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 2u);

  auto expected = specfem::datetime::make(2003, 12, 26, 1, 56, 52.4);

  ASSERT_TRUE(sources[0]->get_starttime().has_value());
  ASSERT_TRUE(sources[1]->get_starttime().has_value());
  EXPECT_EQ(*sources[0]->get_starttime(), expected);
  EXPECT_EQ(*sources[1]->get_starttime(), expected);
}

TEST(ReadSourcesDatetime3D, PerSourceDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim3,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim3/sources_per_source_datetime.yaml"),
          nsteps, dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 2u);

  auto expected_0 = specfem::datetime::make(2003, 12, 26, 1, 56, 50.0);
  auto expected_1 = specfem::datetime::make(2003, 12, 26, 1, 56, 52.0);

  ASSERT_TRUE(sources[0]->get_starttime().has_value());
  ASSERT_TRUE(sources[1]->get_starttime().has_value());
  EXPECT_EQ(*sources[0]->get_starttime(), expected_0);
  EXPECT_EQ(*sources[1]->get_starttime(), expected_1);
}

TEST(ReadSourcesDatetime3D, NoDatetime) {
  auto sources =
      specfem::io::sources_impl::read<specfem::element::dimension_tag::dim3,
                                      specfem::enums::source_format::YAML>(
          std::string("io/sources/data/dim3/single_force.yaml"), nsteps, dt,
          specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 1u);
  EXPECT_FALSE(sources[0]->get_starttime().has_value());
}

// ---------------------------------------------------------------------------
// CMTSOLUTION / FORCESOLUTION datetime tests
// ---------------------------------------------------------------------------

TEST(ReadSourcesDatetime3D, CMTSOLUTIONDatetime) {
  auto sources = specfem::io::sources_impl::read<
      specfem::element::dimension_tag::dim3,
      specfem::enums::source_format::CMTSOLUTION>(
      std::string("io/sources/data/dim3/single_moment_tensor.CMTSOLUTION"),
      nsteps, dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 1u);
  ASSERT_TRUE(sources[0]->get_starttime().has_value());
  auto expected = specfem::datetime::make(2000, 1, 1, 0, 0, 0.0);
  EXPECT_EQ(*sources[0]->get_starttime(), expected);
}

TEST(ReadSourcesDatetime3D, FORCESOLUTIONNoDatetime) {
  auto sources = specfem::io::sources_impl::read<
      specfem::element::dimension_tag::dim3,
      specfem::enums::source_format::FORCESOLUTION>(
      std::string("io/sources/data/dim3/single_force.FORCESOLUTION"), nsteps,
      dt, specfem::simulation::field_type::forward);

  ASSERT_EQ(sources.size(), 1u);
  EXPECT_FALSE(sources[0]->get_starttime().has_value());
}
