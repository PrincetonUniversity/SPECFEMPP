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
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>> {
  source_parameters()
      : x(0.0), y(0.0), z(0.0), Mcxx(0.0), Mcyy(0.0), Mczz(0.0), Mcxy(0.0),
        Mcxz(0.0), Mcyz(0.0), Mcyx(0.0), Mczx(0.0), Mczy(0.0) {};
  // Symmetric overload: lower triangle defaults to the upper triangle.
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real Mcxx, type_real Mcyy, type_real Mczz,
                    type_real Mcxy, type_real Mcxz, type_real Mcyz,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz),
        Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz), Mcyx(Mcxy), Mczx(Mcxz), Mczy(Mcyz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};
  // Asymmetric overload: lower triangle given explicitly.
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real Mcxx, type_real Mcyy, type_real Mczz,
                    type_real Mcxy, type_real Mcxz, type_real Mcyz,
                    type_real Mcyx, type_real Mczx, type_real Mczy,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz),
        Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz), Mcyx(Mcyx), Mczx(Mczx), Mczy(Mczy),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real y;      ///< y-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mcxx;   ///< Mcxx component of spin tensor
  type_real Mcyy;   ///< Mcyy component of spin tensor
  type_real Mczz;   ///< Mczz component of spin tensor
  type_real Mcxy;   ///< Mcxy component of spin tensor
  type_real Mcxz;   ///< Mcxz component of spin tensor
  type_real Mcyz;   ///< Mcyz component of spin tensor
  type_real Mcyx;   ///< Mcyx component of spin tensor (defaults to Mcxy)
  type_real Mczx;   ///< Mczx component of spin tensor (defaults to Mcxz)
  type_real Mczy;   ///< Mczy component of spin tensor (defaults to Mcyz)
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag;        ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>> {
public:
  source_solution(type_real x, type_real y, type_real z,
                  std::vector<std::vector<type_real>> source_tensor)
      : x(x), y(y), z(z) {
    this->source_tensor =
        Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>(
            "source_tensor", source_tensor.size(), source_tensor[0].size());
    for (size_t i = 0; i < source_tensor.size(); ++i) {
      for (size_t j = 0; j < source_tensor[i].size(); ++j) {
        this->source_tensor(i, j) = source_tensor[i][j];
      }
    }
  }

  type_real x; ///< x-coordinate of the source
  type_real y; ///< y-coordinate of the source
  type_real z; ///< z-coordinate of the source
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_tensor; ///< Expected source tensor values
};

// Defining short hands for the source parameters and solution types
using SpinTensorSource3DSolution = source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>>;
using SpinTensorSource3DParameters = source_parameters<
    specfem::element::dimension_tag::dim3,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>>;

using SpinTensorSource3DParametersAndSolution =
    std::tuple<SpinTensorSource3DParameters, SpinTensorSource3DSolution>;
// Vector of pairs of spin tensor parameters and corresponding tensor solutions
template <>
std::vector<SpinTensorSource3DParametersAndSolution>
get_parameters_and_solutions<
    specfem::element::dimension_tag::dim3,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>>() {
  return std::vector<SpinTensorSource3DParametersAndSolution>{
    // Symmetric elastic_spin spin tensor: displacement rows zero, rotation rows
    // carry the symmetric tensor.
    std::make_tuple(
        SpinTensorSource3DParameters(
            "3D elastic_spin", 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        SpinTensorSource3DSolution(
            2.0, 2.0, 2.0,
            std::vector<std::vector<type_real>>{ { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 1.0, 0.5, 0.6 },
                                                 { 0.5, 2.0, 0.7 },
                                                 { 0.6, 0.7, 3.0 } })),
    // Asymmetric elastic_spin spin tensor.
    std::make_tuple(
        SpinTensorSource3DParameters(
            "3D elastic_spin asymmetric", 2.0, 2.0, 2.0, 1.0, 2.0, 3.0, 0.5,
            0.6, 0.7, -0.5, -0.6, -0.7,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        SpinTensorSource3DSolution(
            2.0, 2.0, 2.0,
            std::vector<std::vector<type_real>>{ { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 0.0, 0.0, 0.0 },
                                                 { 1.0, 0.5, 0.6 },
                                                 { -0.5, 2.0, 0.7 },
                                                 { -0.6, -0.7, 3.0 } })),
  };
}

// Factory function specialization for 3D Spin Tensor Source. Always uses the
// asymmetric (12-arg) constructor; for symmetric parameters the lower triangle
// equals the upper triangle.
template <>
specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>
create_source<
    specfem::element::dimension_tag::dim3,
    specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>>(
    const source_parameters<
        specfem::element::dimension_tag::dim3,
        specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>>
        &parameters) {
  return specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>(
      parameters.x, parameters.y, parameters.z, parameters.Mcxx,
      parameters.Mcyy, parameters.Mczz, parameters.Mcxy, parameters.Mcxz,
      parameters.Mcyz, parameters.Mcyx, parameters.Mczx, parameters.Mczy,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      parameters.wavefield_type);
}

// Spin tensor specific tests (issue #2113)

// Helper macro to declare a 3D spin tensor local `name` with the given
// components and medium tag. The source base class is non-copyable (holds a
// unique_ptr), so sources must be constructed in place rather than returned.
#define MAKE_SPIN_TENSOR_3D(name, Mcxx, Mcyy, Mczz, Mcxy, Mcxz, Mcyz, Mcyx,    \
                            Mczx, Mczy, medium)                                \
  specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3> name(   \
      0.0, 0.0, 0.0, (Mcxx), (Mcyy), (Mczz), (Mcxy), (Mcxz), (Mcyz), (Mcyx),   \
      (Mczx), (Mczy),                                                          \
      std::make_unique<specfem::source_time_functions::Ricker>(                \
          10, 0.01, 1.0, 0.0, 1.0, false),                                     \
      specfem::simulation::field_type::forward);                               \
  name.set_medium_tag(medium)

// A spin tensor never produces a body-couple (monopole) contribution
// (issue #2113).
TEST(SPIN_TENSOR_3D, NoMonopoleContribution) {
  MAKE_SPIN_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                      specfem::element::medium_tag::elastic_spin);
  EXPECT_FALSE(source.has_monopole_contribution());
  EXPECT_EQ(source.get_body_couple_vector().extent(0), 0u);
}

// Construction and getters (issue #2113).
TEST(SPIN_TENSOR_3D, ConstructionAndGetters) {
  MAKE_SPIN_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                      specfem::element::medium_tag::elastic_spin);
  EXPECT_NEAR(source.get_Mcxx(), 1.0, 1e-6);
  EXPECT_NEAR(source.get_Mcyy(), 2.0, 1e-6);
  EXPECT_NEAR(source.get_Mczz(), 3.0, 1e-6);
  EXPECT_NEAR(source.get_Mcxy(), 0.5, 1e-6);
  EXPECT_NEAR(source.get_Mcxz(), 0.6, 1e-6);
  EXPECT_NEAR(source.get_Mcyz(), 0.7, 1e-6);
  EXPECT_NEAR(source.get_Mcyx(), -0.5, 1e-6);
  EXPECT_NEAR(source.get_Mczx(), -0.6, 1e-6);
  EXPECT_NEAR(source.get_Mczy(), -0.7, 1e-6);
  EXPECT_EQ(source.source_name(), "3-D spin tensor");
  EXPECT_EQ(source.get_supported_media(),
            std::vector<specfem::element::medium_tag>{
                specfem::element::medium_tag::elastic_spin });
}

// The symmetric (9-arg) ctor defaults the lower triangle to the transpose.
TEST(SPIN_TENSOR_3D, SymmetricCtorDefaultsLowerTriangle) {
  specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3> source(
      0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      specfem::simulation::field_type::forward);
  EXPECT_NEAR(source.get_Mcyx(), 0.5, 1e-6);
  EXPECT_NEAR(source.get_Mczx(), 0.6, 1e-6);
  EXPECT_NEAR(source.get_Mczy(), 0.7, 1e-6);
}

// print_details reports the Mc components (issue #2113).
TEST(SPIN_TENSOR_3D, PrintDetailsIncludesComponents) {
  MAKE_SPIN_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                      specfem::element::medium_tag::elastic_spin);
  const std::string details = source.print_details();
  EXPECT_NE(details.find("Mcxx"), std::string::npos);
  EXPECT_NE(details.find("Mcyx"), std::string::npos);
  EXPECT_NE(details.find("Mczy"), std::string::npos);
}

// Equality compares the Mc components and distinguishes spin tensors from
// moment tensors (issue #2113).
TEST(SPIN_TENSOR_3D, OperatorEq) {
  MAKE_SPIN_TENSOR_3D(source, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                      specfem::element::medium_tag::elastic_spin);
  MAKE_SPIN_TENSOR_3D(same, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, -0.5, -0.6, -0.7,
                      specfem::element::medium_tag::elastic_spin);
  MAKE_SPIN_TENSOR_3D(different, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7,
                      specfem::element::medium_tag::elastic_spin);
  EXPECT_TRUE(source == same);
  EXPECT_TRUE(source != different);

  // A moment tensor is never equal to a spin tensor, even with matching
  // component values. Compare through the base reference to avoid the C++20
  // rewritten-operator ambiguity between two distinct derived types.
  specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>
      moment_tensor_source(
          0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
          std::make_unique<specfem::source_time_functions::Ricker>(
              10, 0.01, 1.0, 0.0, 1.0, false),
          specfem::simulation::field_type::forward);
  moment_tensor_source.set_medium_tag(
      specfem::element::medium_tag::elastic_spin);
  const specfem::sources::source<specfem::element::dimension_tag::dim3>
      &moment_tensor_ref = moment_tensor_source;
  EXPECT_TRUE(source != moment_tensor_ref);
}

#undef MAKE_SPIN_TENSOR_3D

// Explicit template instantiations
