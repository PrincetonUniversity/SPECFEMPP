#include "../../source.hpp"

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

template <>
struct source_parameters<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>> {
  source_parameters()
      : x(0.0), y(0.0), z(0.0), Mxx(0.0), Myy(0.0), Mzz(0.0), Mxy(0.0),
        Mxz(0.0), Myz(0.0), Myx(0.0), Mzx(0.0), Mzy(0.0) {};
  // Symmetric overload: lower triangle defaults to the upper triangle.
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real Mxx, type_real Myy, type_real Mzz, type_real Mxy,
                    type_real Mxz, type_real Myz,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy),
        Mxz(Mxz), Myz(Myz), Myx(Mxy), Mzx(Mxz), Mzy(Myz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};
  // Asymmetric overload: lower triangle given explicitly.
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real Mxx, type_real Myy, type_real Mzz, type_real Mxy,
                    type_real Mxz, type_real Myz, type_real Myx, type_real Mzx,
                    type_real Mzy,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy),
        Mxz(Mxz), Myz(Myz), Myx(Myx), Mzx(Mzx), Mzy(Mzy),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real y;      ///< y-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mxx;    ///< Mxx component of moment tensor
  type_real Myy;    ///< Myy component of moment tensor
  type_real Mzz;    ///< Mzz component of moment tensor
  type_real Mxy;    ///< Mxy component of moment tensor
  type_real Mxz;    ///< Mxz component of moment tensor
  type_real Myz;    ///< Myz component of moment tensor
  type_real Myx;    ///< Myx component of moment tensor (defaults to Mxy)
  type_real Mzx;    ///< Mzx component of moment tensor (defaults to Mxz)
  type_real Mzy;    ///< Mzy component of moment tensor (defaults to Myz)
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag;        ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>> {
public:
  source_solution(type_real x, type_real y, type_real z, type_real Mxx,
                  type_real Myy, type_real Mzz, type_real Mxy, type_real Mxz,
                  type_real Myz,
                  std::vector<std::vector<type_real>> source_tensor)
      : x(x), y(y), z(z), Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy), Mxz(Mxz),
        Myz(Myz) {
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
  type_real y;   ///< y-coordinate of the source
  type_real z;   ///< z-coordinate of the source
  type_real Mxx; ///< Mxx component of moment tensor
  type_real Myy; ///< Myy component of moment tensor
  type_real Mzz; ///< Mzz component of moment tensor
  type_real Mxy; ///< Mxy component of moment tensor
  type_real Mxz; ///< Mxz component of moment tensor
  type_real Myz; ///< Myz component of moment tensor
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_tensor; ///< Source tensor in
                     ///< Kokkos format
};

// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
// Defining short hands for the source parameters and solution types
using MomentTensorSource3DSolution = source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>;
using MomentTensorSource3DParameters = source_parameters<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>;

using MomentTensorSource3DParametersAndSolution =
    std::tuple<MomentTensorSource3DParameters, MomentTensorSource3DSolution>;
// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
template <>
std::vector<MomentTensorSource3DParametersAndSolution>
get_parameters_and_solutions<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>() {
  return std::vector<MomentTensorSource3DParametersAndSolution>{
    // Test 3D elastic moment tensor source (simple diagonal)
    std::make_tuple(
        MomentTensorSource3DParameters("3D elastic diagonal", 0.0, 0.0, 0.0,
                                       1.0, 2.0, 3.0, 0.0, 0.0, 0.0,
                                       specfem::simulation::field_type::forward,
                                       specfem::element::medium_tag::elastic),
        MomentTensorSource3DSolution(
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0,
            std::vector<std::vector<type_real>>{
                { 1.0, 0.0, 0.0 }, { 0.0, 2.0, 0.0 }, { 0.0, 0.0, 3.0 } })),
    // Test 3D elastic moment tensor source (full tensor)
    std::make_tuple(
        MomentTensorSource3DParameters("3D elastic full", 1.0, 1.0, 1.0, 1.0,
                                       2.0, 3.0, 0.5, 0.6, 0.7,
                                       specfem::simulation::field_type::forward,
                                       specfem::element::medium_tag::elastic),
        MomentTensorSource3DSolution(
            1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            std::vector<std::vector<type_real>>{
                { 1.0, 0.5, 0.6 }, { 0.5, 2.0, 0.7 }, { 0.6, 0.7, 3.0 } })),
    // Test 3D elastic_spin (Cosserat) moment tensor source. The three
    // displacement rows carry the symmetric tensor; the three rotation rows are
    // zero (rotation enters via the body couple).
    std::make_tuple(
        MomentTensorSource3DParameters(
            "3D elastic_spin", 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        MomentTensorSource3DSolution(
            2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            std::vector<std::vector<type_real>>{ { 1.0, 0.5, 0.6 },
                                                 { 0.5, 2.0, 0.7 },
                                                 { 0.6, 0.7, 3.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 } })),
    // Test 3D asymmetric elastic_spin moment tensor. The lower-triangle
    // components (Myx, Mzx, Mzy) populate the corresponding displacement-row
    // slots; the rotation rows remain zero (driven by the body couple).
    std::make_tuple(
        MomentTensorSource3DParameters(
            "3D elastic_spin asymmetric", 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 0.5,
            0.6, 0.7, -0.5, -0.6, -0.7,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        MomentTensorSource3DSolution(
            2.0, 2.0, 2.0, 1.0, 2.0,
            3.0, 0.5, 0.6, 0.7,
            std::vector<std::vector<type_real>>{ { 1.0, 0.5, 0.6 },
                                                 { -0.5, 2.0, 0.7 },
                                                 { -0.6, -0.7, 3.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 } })),
  };
}

// Factory function specialization for 3D Moment Tensor Source. Always uses the
// asymmetric (12-arg) constructor; for symmetric parameters the lower triangle
// equals the upper triangle, reproducing the symmetric tensor.
template <>
specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
create_source<
    specfem::element::dimension_tag::dim3,
    specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>(
    const source_parameters<
        specfem::element::dimension_tag::dim3,
        specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>>
        &parameters) {
  return specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>(
      parameters.x, parameters.y, parameters.z, parameters.Mxx, parameters.Myy,
      parameters.Mzz, parameters.Mxy, parameters.Mxz, parameters.Myz,
      parameters.Myx, parameters.Mzx, parameters.Mzy,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      parameters.wavefield_type);
}

// Body couple tests for 3D moment tensor sources (issue #2113)

// Helper macro to declare a 3D moment tensor local `name` with the given
// components and medium tag. The source base class is non-copyable (holds a
// unique_ptr), so sources must be constructed in place rather than returned.
#define MAKE_MOMENT_TENSOR_3D(name, Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx,    \
                              Mzy, medium)                                     \
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3> name( \
      0.0, 0.0, 0.0, (Mxx), (Myy), (Mzz), (Mxy), (Mxz), (Myz), (Myx), (Mzx),   \
      (Mzy),                                                                   \
      std::make_unique<specfem::source_time_functions::Ricker>(                \
          10, 0.01, 1.0, 0.0, 1.0, false),                                     \
      specfem::simulation::field_type::forward);                               \
  name.set_medium_tag(medium)

// An asymmetric moment tensor on elastic_spin yields a body couple
// [0, 0, 0, Myz - Mzy, Mzx - Mxz, Mxy - Myx] (issue #2113).
TEST(MOMENT_TENSOR_BODY_COUPLE_3D, AsymmetricSpin) {
  const type_real Mxy = 0.5, Mxz = 0.6, Myz = 0.7;
  const type_real Myx = -0.5, Mzx = -0.6, Mzy = -0.7;
  MAKE_MOMENT_TENSOR_3D(source, 1.0, 2.0, 3.0, Mxy, Mxz, Myz, Myx, Mzx, Mzy,
                        specfem::element::medium_tag::elastic_spin);

  auto body_couple = source.get_body_couple_vector();
  ASSERT_EQ(body_couple.extent(0), 6u);
  EXPECT_NEAR(body_couple(0), 0.0, 1e-6);
  EXPECT_NEAR(body_couple(1), 0.0, 1e-6);
  EXPECT_NEAR(body_couple(2), 0.0, 1e-6);
  EXPECT_NEAR(body_couple(3), Myz - Mzy, 1e-5); // (eps:M)_x
  EXPECT_NEAR(body_couple(4), Mzx - Mxz, 1e-5); // (eps:M)_y
  EXPECT_NEAR(body_couple(5), Mxy - Myx, 1e-5); // (eps:M)_z
}

// A symmetric moment tensor produces zero rotational coupling (issue #2113).
TEST(MOMENT_TENSOR_BODY_COUPLE_3D, SymmetricIsZero) {
  // Via the 12-arg ctor with transpose-equal values.
  MAKE_MOMENT_TENSOR_3D(explicit_symmetric, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.5,
                        0.6, 0.7, specfem::element::medium_tag::elastic_spin);
  auto bc = explicit_symmetric.get_body_couple_vector();
  EXPECT_NEAR(bc(3), 0.0, 1e-6);
  EXPECT_NEAR(bc(4), 0.0, 1e-6);
  EXPECT_NEAR(bc(5), 0.0, 1e-6);

  // Via the 9-arg (symmetric) ctor: lower triangle defaults to the transpose.
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
      defaulted(0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
                std::make_unique<specfem::source_time_functions::Ricker>(
                    10, 0.01, 1.0, 0.0, 1.0, false),
                specfem::simulation::field_type::forward);
  defaulted.set_medium_tag(specfem::element::medium_tag::elastic_spin);
  auto bc_def = defaulted.get_body_couple_vector();
  EXPECT_NEAR(bc_def(3), 0.0, 1e-6);
  EXPECT_NEAR(bc_def(4), 0.0, 1e-6);
  EXPECT_NEAR(bc_def(5), 0.0, 1e-6);
}

// Non-Cosserat media have no rotational degree of freedom and therefore an
// empty body couple (issue #2113).
TEST(MOMENT_TENSOR_BODY_COUPLE_3D, EmptyForNonCosseratMedia) {
  MAKE_MOMENT_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                        specfem::element::medium_tag::elastic);
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);
  EXPECT_FALSE(source.has_monopole_contribution());
}

// print_details reports the lower-triangle components as part of a single
// source entry (issue #2113).
TEST(MOMENT_TENSOR_BODY_COUPLE_3D, PrintDetailsIncludesAsymmetric) {
  MAKE_MOMENT_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                        specfem::element::medium_tag::elastic_spin);
  const std::string details = source.print_details();
  EXPECT_NE(details.find("Myx"), std::string::npos);
  EXPECT_NE(details.find("Mzx"), std::string::npos);
  EXPECT_NE(details.find("Mzy"), std::string::npos);
}

// Equality distinguishes the lower-triangle components, and a defaulted lower
// triangle equals an explicit symmetric one (issue #2113).
TEST(MOMENT_TENSOR_BODY_COUPLE_3D, OperatorEqDistinguishesAsymmetric) {
  MAKE_MOMENT_TENSOR_3D(asymmetric, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6,
                        -0.7, specfem::element::medium_tag::elastic_spin);
  MAKE_MOMENT_TENSOR_3D(symmetric, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7,
                        specfem::element::medium_tag::elastic_spin);
  EXPECT_TRUE(asymmetric != symmetric);

  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
      defaulted(0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
                std::make_unique<specfem::source_time_functions::Ricker>(
                    10, 0.01, 1.0, 0.0, 1.0, false),
                specfem::simulation::field_type::forward);
  defaulted.set_medium_tag(specfem::element::medium_tag::elastic_spin);
  EXPECT_TRUE(defaulted == symmetric);
}

#undef MAKE_MOMENT_TENSOR_3D

// Explicit template instantiations
