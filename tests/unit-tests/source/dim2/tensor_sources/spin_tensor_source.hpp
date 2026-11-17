
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>

template <>
struct source_parameters<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>> {
  source_parameters() : x(0.0), z(0.0), Mcyx(0.0), Mcyz(0.0) {};
  source_parameters(std::string name, type_real x, type_real z, type_real Mcyx,
                    type_real Mcyz,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), Mcyx(Mcyx), Mcyz(Mcyz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mcyx;   ///< Mcyx component of spin tensor
  type_real Mcyz;   ///< Mcyz component of spin tensor
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag;        ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>> {
public:
  source_solution(type_real x, type_real z, type_real Mcyx, type_real Mcyz,
                  std::vector<std::vector<type_real>> source_tensor)
      : x(x), z(z), Mcyx(Mcyx), Mcyz(Mcyz) {
    this->source_tensor =
        Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>(
            "source_tensor", source_tensor.size(), source_tensor[0].size());
    for (size_t i = 0; i < source_tensor.size(); ++i) {
      for (size_t j = 0; j < source_tensor[i].size(); ++j) {
        this->source_tensor(i, j) = source_tensor[i][j];
      }
    }
  }

  type_real x;    ///< x-coordinate of the source
  type_real z;    ///< z-coordinate of the source
  type_real Mcyx; ///< Mcyx component of spin tensor
  type_real Mcyz; ///< Mcyz component of spin tensor
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_tensor; ///< Expected source tensor values
};

// Defining short hands for the source parameters and solution types
using SpinTensorSource2DSolution = source_solution<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>>;
using SpinTensorSource2DParameters = source_parameters<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>>;

using SpinTensorSource2DParametersAndSolution =
    std::tuple<SpinTensorSource2DParameters, SpinTensorSource2DSolution>;
// Vector of pairs of spin tensor parameters and corresponding tensor solutions
template <>
std::vector<SpinTensorSource2DParametersAndSolution>
get_parameters_and_solutions<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>>() {
  return std::vector<SpinTensorSource2DParametersAndSolution>{
    // Test elastic P-SV-T spin tensor source: the displacement rows are zero
    // and the rotation row carries [Mcyx, Mcyz].
    std::make_tuple(SpinTensorSource2DParameters(
                        "elastic_psv_t", 2.0, 2.0, 0.8, -0.3,
                        specfem::simulation::field_type::forward,
                        specfem::element::medium_tag::elastic_psv_t),
                    SpinTensorSource2DSolution(
                        2.0, 2.0, 0.8, -0.3,
                        std::vector<std::vector<type_real>>{
                            { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.8, -0.3 } }))
  };
}

// Factory function specialization for 2D Spin Tensor Source
template <>
specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>
create_source<
    specfem::element::dimension_tag::dim2,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>>(
    const source_parameters<
        specfem::element::dimension_tag::dim2,
        specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>>
        &parameters) {
  return specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>(
      parameters.x, parameters.z, parameters.Mcyx, parameters.Mcyz,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      parameters.wavefield_type);
}

// Spin tensor specific tests (issue #2112)

// Helper macro to declare a 2D spin tensor local `name` with the given
// components and medium tag. The source base class is non-copyable (holds a
// unique_ptr), so sources must be constructed in place rather than returned.
#define MAKE_SPIN_TENSOR(name, Mcyx, Mcyz, medium)                             \
  specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2> name(   \
      0.0, 0.0, (Mcyx), (Mcyz),                                                \
      std::make_unique<specfem::source_time_functions::Ricker>(                \
          10, 0.01, 1.0, 0.0, 1.0, false),                                     \
      specfem::simulation::field_type::forward);                               \
  name.set_medium_tag(medium)

// A spin tensor never produces a body-couple (monopole) contribution, no
// matter its components (issue #2112).
TEST(SPIN_TENSOR, NoMonopoleContribution) {
  MAKE_SPIN_TENSOR(source, 0.8, -0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  EXPECT_FALSE(source.has_monopole_contribution());
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);
}

// Construction and getters (issue #2112).
TEST(SPIN_TENSOR, ConstructionAndGetters) {
  MAKE_SPIN_TENSOR(source, 0.8, -0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  EXPECT_NEAR(source.get_Mcyx(), 0.8, 1e-6);
  EXPECT_NEAR(source.get_Mcyz(), -0.3, 1e-6);
  EXPECT_EQ(source.source_name(), "2-D spin tensor");
  EXPECT_EQ(source.get_supported_media(),
            std::vector<specfem::element::medium_tag>{
                specfem::element::medium_tag::elastic_psv_t });
}

// print_details reports the Mc components (issue #2112).
TEST(SPIN_TENSOR, PrintDetailsIncludesComponents) {
  MAKE_SPIN_TENSOR(source, 0.8, -0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  const std::string details = source.print_details();
  EXPECT_NE(details.find("Mcyx"), std::string::npos);
  EXPECT_NE(details.find("Mcyz"), std::string::npos);
}

// Equality compares the Mc components and distinguishes spin tensors from
// moment tensors (issue #2112).
TEST(SPIN_TENSOR, OperatorEq) {
  MAKE_SPIN_TENSOR(source, 0.8, -0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  MAKE_SPIN_TENSOR(same, 0.8, -0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  MAKE_SPIN_TENSOR(different, 0.8, 0.3,
                   specfem::element::medium_tag::elastic_psv_t);
  EXPECT_TRUE(source == same);
  EXPECT_TRUE(source != different);

  // A moment tensor is never equal to a spin tensor, even with matching
  // component values.
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
      moment_tensor_source(
          0.0, 0.0, 0.8, -0.3, 0.0,
          std::make_unique<specfem::source_time_functions::Ricker>(
              10, 0.01, 1.0, 0.0, 1.0, false),
          specfem::simulation::field_type::forward);
  moment_tensor_source.set_medium_tag(
      specfem::element::medium_tag::elastic_psv_t);
  // Compare through the base reference: comparing two distinct derived types
  // directly is ambiguous under C++20 rewritten-operator rules.
  const specfem::sources::source<specfem::element::dimension_tag::dim2>
      &moment_tensor_ref = moment_tensor_source;
  EXPECT_TRUE(source != moment_tensor_ref);
}

#undef MAKE_SPIN_TENSOR
