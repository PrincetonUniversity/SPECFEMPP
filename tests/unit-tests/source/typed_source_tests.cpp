#include "source.hpp"
#include "source/dim2/tensor_sources/moment_tensor_source.hpp"
#include "source/dim2/vector_sources/adjoint_source.hpp"
#include "source/dim2/vector_sources/cosserat_force_source.hpp"
#include "source/dim2/vector_sources/external_source.hpp"
#include "source/dim2/vector_sources/force_source.hpp"
#include "source/dim3/tensor_sources/moment_tensor_source.hpp"
#include "source/dim3/vector_sources/force_source.hpp"
#include <gtest/gtest.h>

// Type definitions for 2D sources
using SourceTypes = ::testing::Types<
    specfem::sources::force<specfem::dimension::type::dim2>,
    specfem::sources::external<specfem::dimension::type::dim2>,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2>,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2>,
    specfem::sources::moment_tensor<specfem::dimension::type::dim2>,
    specfem::sources::force<specfem::dimension::type::dim3>,
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> >;

// Template test fixture for all sources
template <typename T> class SourceTest : public ::testing::Test {
protected:
  using SourceType = T;
  // Extract dimension tag from the source type
  static constexpr auto DimensionTag = T::dimension_tag;
};

// Test name generators
struct SourceTestNameGenerator {
  template <typename T> static std::string GetName(int) {
    return std::string(T::name);
  }
};

// Register the typed test suites with name generators
TYPED_TEST_SUITE(SourceTest, SourceTypes);

// Source typed tests, Source Parameters and Solutions (SPS)
TYPED_TEST(SourceTest, Type) {
  using SourceType = typename TestFixture::SourceType;
  constexpr auto DimensionTag = TestFixture::DimensionTag;

  auto parameters_and_solutions =
      get_parameters_and_solutions<DimensionTag, SourceType>();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &parameters = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);
    test_source<DimensionTag, SourceType>(parameters, solution);
  }
}
