#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct moment_tensor_parameters {
  moment_tensor_parameters() : x(0.0), z(0.0), Mxx(0.0), Mzz(0.0), Mxz(0.0) {};
  moment_tensor_parameters(std::string name, type_real x, type_real z,
                           type_real Mxx, type_real Mzz, type_real Mxz,
                           specfem::wavefield::simulation_field wavefield_type,
                           specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mxx;    ///< Mxx component of moment tensor
  type_real Mzz;    ///< Mzz component of moment tensor
  type_real Mxz;    ///< Mxz component of moment tensor
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

struct moment_tensor_solution {
public:
  moment_tensor_solution(type_real x, type_real z, type_real Mxx, type_real Mzz,
                         type_real Mxz,
                         std::vector<std::vector<type_real> > source_tensor)
      : x(x), z(z), Mxx(Mxx), Mzz(Mzz), Mxz(Mxz) {
    this->source_tensor = specfem::kokkos::HostView2d<type_real>(
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
  specfem::kokkos::HostView2d<type_real> source_tensor; ///< Source tensor in
                                                        ///< Kokkos format
};

// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
std::vector<std::tuple<moment_tensor_parameters, moment_tensor_solution> >
get_moment_tensor_parameters_and_solutions() {
  return std::vector<
      std::tuple<moment_tensor_parameters, moment_tensor_solution> >{
    // Test elastic P-SV moment tensor source
    std::make_tuple(
        moment_tensor_parameters("elastic_psv", 0.0, 0.0, 1.0, 2.0, 0.5,
                                 specfem::wavefield::simulation_field::forward,
                                 specfem::element::medium_tag::elastic_psv),
        moment_tensor_solution(0.0, 0.0, 1.0, 2.0, 0.5,
                               std::vector<std::vector<type_real> >{
                                   { 1.0, 0.5 }, { 0.5, 2.0 } })),
    // Test poroelastic moment tensor source (elastic tensor repeated twice)
    std::make_tuple(
        moment_tensor_parameters("poroelastic", 1.0, 1.0, 2.0, 3.0, 1.5,
                                 specfem::wavefield::simulation_field::forward,
                                 specfem::element::medium_tag::poroelastic),
        moment_tensor_solution(
            1.0, 1.0, 2.0, 3.0, 1.5,
            std::vector<std::vector<type_real> >{
                { 2.0, 1.5 }, { 1.5, 3.0 }, { 2.0, 1.5 }, { 1.5, 3.0 } })),
    // Test elastic P-SV-T moment tensor source (third component zero)
    std::make_tuple(
        moment_tensor_parameters("elastic_psv_t", 2.0, 2.0, 0.8, 1.2, 0.3,
                                 specfem::wavefield::simulation_field::forward,
                                 specfem::element::medium_tag::elastic_psv_t),
        moment_tensor_solution(2.0, 2.0, 0.8, 1.2, 0.3,
                               std::vector<std::vector<type_real> >{
                                   { 0.8, 0.3 }, { 0.3, 1.2 }, { 0.0, 0.0 } })),
    // Test electromagnetic TE moment tensor source
    std::make_tuple(moment_tensor_parameters(
                        "electromagnetic_te", 3.0, 3.0, 1.5, 2.5, 0.8,
                        specfem::wavefield::simulation_field::forward,
                        specfem::element::medium_tag::electromagnetic_te),
                    moment_tensor_solution(3.0, 3.0, 1.5, 2.5, 0.8,
                                           std::vector<std::vector<type_real> >{
                                               { 1.5, 0.8 }, { 0.8, 2.5 } }))
  };
}

// Now for each parameter set, we will create a moment tensor source and compute
// the source tensor using the get_source_tensor method.

TEST(SOURCES, moment_tensor_source_tensor) {
  // Get the parameters and solutions
  auto parameters_and_solutions = get_moment_tensor_parameters_and_solutions();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &params = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);

    // Add to trace for easier debugging
    SCOPED_TRACE("Testing moment tensor source for: " + params.name);

    // Create a moment tensor source
    specfem::sources::moment_tensor<specfem::dimension::type::dim2>
        moment_tensor_source(
            params.x, params.z, params.Mxx, params.Mzz, params.Mxz,
            std::make_unique<specfem::forcing_function::Ricker>(
                10, 0.01, 1.0, 0.0, 1.0, false),
            params.wavefield_type);

    // Set the medium tag
    moment_tensor_source.set_medium_tag(params.medium_tag);

    // Compare x, z, and moment tensor components (this is just reassignment, no
    // computation)
    EXPECT_REAL_EQ(params.x, moment_tensor_source.get_x());
    EXPECT_REAL_EQ(params.z, moment_tensor_source.get_z());
    EXPECT_REAL_EQ(params.Mxx, moment_tensor_source.get_Mxx());
    EXPECT_REAL_EQ(params.Mzz, moment_tensor_source.get_Mzz());
    EXPECT_REAL_EQ(params.Mxz, moment_tensor_source.get_Mxz());

    // Get the computed source tensor
    auto computed_source_tensor = moment_tensor_source.get_source_tensor();

    // Get the source tensor from the solution
    auto expected_source_tensor = solution.source_tensor;

    // Check if the computed source tensor matches the expected one
    EXPECT_EQ(computed_source_tensor.extent(0),
              expected_source_tensor.extent(0));
    EXPECT_EQ(computed_source_tensor.extent(1),
              expected_source_tensor.extent(1));

    // Compare each component of the source tensor
    for (size_t i = 0; i < expected_source_tensor.extent(0); ++i) {
      for (size_t j = 0; j < expected_source_tensor.extent(1); ++j) {
        EXPECT_NEAR(computed_source_tensor(i, j), expected_source_tensor(i, j),
                    1e-5)
            << "Mismatch at index (" << i << ", " << j << ")";
      }
    }
  }
}
