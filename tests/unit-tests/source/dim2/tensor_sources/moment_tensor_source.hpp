
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
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>> {
  source_parameters()
      : x(0.0), z(0.0), Mxx(0.0), Mzz(0.0), Mxz(0.0), Mzx(0.0) {};
  // Symmetric overload: Mzx defaults to Mxz.
  source_parameters(std::string name, type_real x, type_real z, type_real Mxx,
                    type_real Mzz, type_real Mxz,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mxz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};
  // Asymmetric overload: Mzx given explicitly.
  source_parameters(std::string name, type_real x, type_real z, type_real Mxx,
                    type_real Mzz, type_real Mxz, type_real Mzx,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mzx),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mxx;    ///< Mxx component of moment tensor
  type_real Mzz;    ///< Mzz component of moment tensor
  type_real Mxz;    ///< Mxz component of moment tensor
  type_real Mzx;    ///< Mzx component of moment tensor (defaults to Mxz)
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag;        ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::element::dimension_tag::dim2,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>> {
public:
  // Symmetric overload: Mzx defaults to Mxz.
  source_solution(type_real x, type_real z, type_real Mxx, type_real Mzz,
                  type_real Mxz,
                  std::vector<std::vector<type_real>> source_tensor)
      : x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mxz) {
    this->source_tensor =
        Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>(
            "source_tensor", source_tensor.size(), source_tensor[0].size());
    for (size_t i = 0; i < source_tensor.size(); ++i) {
      for (size_t j = 0; j < source_tensor[i].size(); ++j) {
        this->source_tensor(i, j) = source_tensor[i][j];
      }
    }
  }

  // Asymmetric overload: Mzx given explicitly.
  source_solution(type_real x, type_real z, type_real Mxx, type_real Mzz,
                  type_real Mxz, type_real Mzx,
                  std::vector<std::vector<type_real>> source_tensor)
      : x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mzx) {
    this->source_tensor =
        Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>(
            "source_tensor", source_tensor.size(), source_tensor[0].size());
    for (size_t i = 0; i < source_tensor.size(); ++i) {
      for (size_t j = 0; j < source_tensor[i].size(); ++j) {
        this->source_tensor(i, j) = source_tensor[i][j];
      }
    }
  }

  type_real x;   ///< x-coordinate of the source
  type_real z;   ///< z-coordinate of the source
  type_real Mxx; ///< Mxx component of moment tensor
  type_real Mzz; ///< Mzz component of moment tensor
  type_real Mxz; ///< Mxz component of moment tensor
  type_real Mzx; ///< Mzx component of moment tensor (defaults to Mxz)
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_tensor; ///< Expected source tensor values
};

// Defining short hands for the source parameters and solution types
using MomentTensorSource2DSolution = source_solution<
    specfem::element::dimension_tag::dim2,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>;
using MomentTensorSource2DParameters = source_parameters<
    specfem::element::dimension_tag::dim2,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>;

using MomentTensorSource2DParametersAndSolution =
    std::tuple<MomentTensorSource2DParameters, MomentTensorSource2DSolution>;
// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
template <>
std::vector<MomentTensorSource2DParametersAndSolution>
get_parameters_and_solutions<
    specfem::element::dimension_tag::dim2,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>() {
  return std::vector<MomentTensorSource2DParametersAndSolution>{
    // Test elastic P-SV moment tensor source

    std::make_tuple(
        MomentTensorSource2DParameters(
            "elastic_psv", 0.0, 0.0, 1.0, 2.0, 0.5,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_psv),
        MomentTensorSource2DSolution(
            0.0, 0.0, 1.0, 2.0, 0.5,
            std::vector<std::vector<type_real>>{ { 1.0, 0.5 }, { 0.5, 2.0 } })),
    // Test poroelastic moment tensor source (elastic tensor repeated twice)
    std::make_tuple(
        MomentTensorSource2DParameters(
            "poroelastic", 1.0, 1.0, 2.0, 3.0, 1.5,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::poroelastic),
        MomentTensorSource2DSolution(
            1.0, 1.0, 2.0, 3.0, 1.5,
            std::vector<std::vector<type_real>>{
                { 2.0, 1.5 }, { 1.5, 3.0 }, { 2.0, 1.5 }, { 1.5, 3.0 } })),
    // Test elastic P-SV-T moment tensor source (third component zero)
    std::make_tuple(MomentTensorSource2DParameters(
                        "elastic_psv_t", 2.0, 2.0, 0.8, 1.2, 0.3,
                        specfem::simulation::field_type::forward,
                        specfem::element::medium_tag::elastic_psv_t),
                    MomentTensorSource2DSolution(
                        2.0, 2.0, 0.8, 1.2, 0.3,
                        std::vector<std::vector<type_real>>{
                            { 0.8, 0.3 }, { 0.3, 1.2 }, { 0.0, 0.0 } })),
    // Test electromagnetic TE moment tensor source
    std::make_tuple(
        MomentTensorSource2DParameters(
            "electromagnetic_te", 3.0, 3.0, 1.5, 2.5, 0.8,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::electromagnetic_te),
        MomentTensorSource2DSolution(
            3.0, 3.0, 1.5, 2.5, 0.8,
            std::vector<std::vector<type_real>>{ { 1.5, 0.8 }, { 0.8, 2.5 } })),
    // Test asymmetric elastic P-SV-T moment tensor (Mxz != Mzx). The (1,0)
    // slot carries Mzx; the rotation row is zero (driven by the body couple).
    std::make_tuple(MomentTensorSource2DParameters(
                        "elastic_psv_t_asymmetric", 2.0, 2.0, 0.8, 1.2, 0.3,
                        -0.7, specfem::simulation::field_type::forward,
                        specfem::element::medium_tag::elastic_psv_t),
                    MomentTensorSource2DSolution(
                        2.0, 2.0, 0.8, 1.2, 0.3,
                        -0.7,
                        std::vector<std::vector<type_real>>{
                            { 0.8, 0.3 }, { -0.7, 1.2 }, { 0.0, 0.0 } })),
    // Test asymmetric elastic P-SV moment tensor (both displacement rows
    // populated with Mzx in the (1,0) slot).
    std::make_tuple(
        MomentTensorSource2DParameters(
            "elastic_psv_asymmetric", 0.0, 0.0, 1.0, 2.0, 0.5, -0.25,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_psv),
        MomentTensorSource2DSolution(0.0, 0.0, 1.0, 2.0, 0.5, -0.25,
                                     std::vector<std::vector<type_real>>{
                                         { 1.0, 0.5 }, { -0.25, 2.0 } }))
  };
}

// Factory function specialization for 2D Moment Tensor Source
template <>
specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
create_source<
    specfem::element::dimension_tag::dim2,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>(
    const source_parameters<
        specfem::element::dimension_tag::dim2,
        specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>>
        &parameters) {
  return specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>(
      parameters.x, parameters.z, parameters.Mxx, parameters.Mzz,
      parameters.Mxz, parameters.Mzx,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      parameters.wavefield_type);
}

// Explicit template instantiations

// Body couple tests for 2D moment tensor sources

// Helper macro to declare a 2D moment tensor local `name` with the given
// components and medium tag. The source base class is non-copyable (holds a
// unique_ptr), so sources must be constructed in place rather than returned.
#define MAKE_MOMENT_TENSOR(name, Mxx, Mzz, Mxz, Mzx, medium)                   \
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2> name( \
      0.0, 0.0, (Mxx), (Mzz), (Mxz), (Mzx),                                    \
      std::make_unique<specfem::source_time_functions::Ricker>(                \
          10, 0.01, 1.0, 0.0, 1.0, false),                                     \
      specfem::simulation::field_type::forward);                               \
  name.set_medium_tag(medium)

// An asymmetric moment tensor on elastic_psv_t yields a body couple
// [0, 0, Mxz - Mzx] (issue #2112).
TEST(MOMENT_TENSOR_BODY_COUPLE, AsymmetricPsvT) {
  MAKE_MOMENT_TENSOR(source, 0.8, 1.2, 0.3, -0.7,
                     specfem::element::medium_tag::elastic_psv_t);

  auto body_couple = source.get_body_couple_vector();
  ASSERT_EQ(body_couple.extent(0), 3u);
  EXPECT_NEAR(body_couple(0), 0.0, 1e-12);
  EXPECT_NEAR(body_couple(1), 0.0, 1e-12);
  EXPECT_NEAR(body_couple(2), 0.3 - (-0.7), 1e-5);
}

// A symmetric moment tensor (Mxz == Mzx) produces zero rotational coupling
// (issue #2112).
TEST(MOMENT_TENSOR_BODY_COUPLE, SymmetricIsZero) {
  // Via the 6-arg ctor with equal values.
  MAKE_MOMENT_TENSOR(explicit_symmetric, 1.0, 2.0, 0.5, 0.5,
                     specfem::element::medium_tag::elastic_psv_t);
  EXPECT_NEAR(explicit_symmetric.get_body_couple_vector()(2), 0.0, 1e-12);

  // Via the 5-arg ctor (Mzx defaults to Mxz).
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
      defaulted(0.0, 0.0, 1.0, 2.0, 0.5,
                std::make_unique<specfem::source_time_functions::Ricker>(
                    10, 0.01, 1.0, 0.0, 1.0, false),
                specfem::simulation::field_type::forward);
  defaulted.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
  EXPECT_NEAR(defaulted.get_body_couple_vector()(2), 0.0, 1e-12);
}

// Non-Cosserat media have no rotational degree of freedom and therefore an
// empty body couple (issue #2112).
TEST(MOMENT_TENSOR_BODY_COUPLE, EmptyForNonCosseratMedia) {
  for (auto medium_tag : { specfem::element::medium_tag::elastic_psv,
                           specfem::element::medium_tag::poroelastic,
                           specfem::element::medium_tag::electromagnetic_te }) {
    MAKE_MOMENT_TENSOR(source, 1.0, 2.0, 0.5, -0.5, medium_tag);
    EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);
  }
}

// print_details reports Mzx as part of a single source entry (issue #2112).
TEST(MOMENT_TENSOR_BODY_COUPLE, PrintDetailsIncludesMzx) {
  MAKE_MOMENT_TENSOR(source, 0.8, 1.2, 0.3, -0.7,
                     specfem::element::medium_tag::elastic_psv_t);
  const std::string details = source.print_details();
  EXPECT_NE(details.find("Mzx"), std::string::npos);
}

// Equality distinguishes Mzx, and a defaulted Mzx equals an explicit Mzx=Mxz
// (issue #2112).
TEST(MOMENT_TENSOR_BODY_COUPLE, OperatorEqDistinguishesMzx) {
  MAKE_MOMENT_TENSOR(asymmetric, 1.0, 2.0, 0.5, -0.5,
                     specfem::element::medium_tag::elastic_psv_t);
  MAKE_MOMENT_TENSOR(symmetric, 1.0, 2.0, 0.5, 0.5,
                     specfem::element::medium_tag::elastic_psv_t);
  EXPECT_TRUE(asymmetric != symmetric);

  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>
      defaulted(0.0, 0.0, 1.0, 2.0, 0.5,
                std::make_unique<specfem::source_time_functions::Ricker>(
                    10, 0.01, 1.0, 0.0, 1.0, false),
                specfem::simulation::field_type::forward);
  defaulted.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
  MAKE_MOMENT_TENSOR(explicit_symmetric, 1.0, 2.0, 0.5, 0.5,
                     specfem::element::medium_tag::elastic_psv_t);
  EXPECT_TRUE(defaulted == explicit_symmetric);
}

#undef MAKE_MOMENT_TENSOR
