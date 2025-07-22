#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct adjoint_parameters {
  adjoint_parameters() : x(0.0), z(0.0) {};
  adjoint_parameters(std::string name, type_real x, type_real z,
                     specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), medium_tag(medium_tag) {};

  std::string name;                        ///< Name of the source
  type_real x;                             ///< x-coordinate of the source
  type_real z;                             ///< z-coordinate of the source
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

struct adjoint_solution {
public:
  adjoint_solution(type_real x, type_real z,
                   std::vector<type_real> force_vector)
      : x(x), z(z) {
    this->force_vector = specfem::kokkos::HostView1d<type_real>(
        "force_vector", force_vector.size());
    for (size_t i = 0; i < force_vector.size(); ++i) {
      this->force_vector(i) = force_vector[i];
    }
  }

  type_real x; ///< x-coordinate of the source
  type_real z; ///< z-coordinate of the source
  specfem::kokkos::HostView1d<type_real> force_vector; ///< Force vector in
                                                       ///< Kokkos format
};

// Vector of pairs of adjoint parameters and corresponding vector force
// solutions
std::vector<std::tuple<adjoint_parameters, adjoint_solution> >
get_adjoint_parameters_and_solutions() {
  return std::vector<std::tuple<adjoint_parameters, adjoint_solution> >{
    // Test acoustic adjoint source
    std::make_tuple(adjoint_parameters("acoustic", 0.0, 0.0,
                                       specfem::element::medium_tag::acoustic),
                    adjoint_solution(0.0, 0.0, std::vector<type_real>{ 1.0 })),
    // Test elastic SH adjoint source
    std::make_tuple(
        adjoint_parameters("elastic_sh", 1.0, 1.0,
                           specfem::element::medium_tag::elastic_sh),
        adjoint_solution(1.0, 1.0, std::vector<type_real>{ 1.0 })),
    // Test elastic P-SV adjoint source
    std::make_tuple(
        adjoint_parameters("elastic_psv", 2.0, 2.0,
                           specfem::element::medium_tag::elastic_psv),
        adjoint_solution(2.0, 2.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test poroelastic adjoint source
    std::make_tuple(
        adjoint_parameters("poroelastic", 3.0, 3.0,
                           specfem::element::medium_tag::poroelastic),
        adjoint_solution(3.0, 3.0,
                         std::vector<type_real>{ 1.0, 1.0, 1.0, 1.0 })),
    // Test elastic P-SV-T adjoint source (third component should be 0)
    std::make_tuple(
        adjoint_parameters("elastic_psv_t", 4.0, 4.0,
                           specfem::element::medium_tag::elastic_psv_t),
        adjoint_solution(4.0, 4.0, std::vector<type_real>{ 1.0, 1.0, 0.0 })),
    // Test electromagnetic TE adjoint source
    std::make_tuple(
        adjoint_parameters("electromagnetic_te", 5.0, 5.0,
                           specfem::element::medium_tag::electromagnetic_te),
        adjoint_solution(5.0, 5.0, std::vector<type_real>{ 1.0, 1.0 }))
  };
}

// Now for each parameter set, we will create an adjoint source and compute the
// force vector using the get_force_vector method.

TEST(SOURCES, adjoint_source_vector) {
  // Get the parameters and solutions
  auto parameters_and_solutions = get_adjoint_parameters_and_solutions();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &params = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);

    // Add to trace for easier debugging
    SCOPED_TRACE("Testing adjoint source for: " + params.name);

    // Create an adjoint source
    specfem::sources::adjoint_source adjoint_source(
        params.x, params.z,
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        "STA", "NET");

    // Set the medium tag
    adjoint_source.set_medium_tag(params.medium_tag);

    // Compare x and z (this is just reassignment, no computation)
    EXPECT_REAL_EQ(params.x, adjoint_source.get_x());
    EXPECT_REAL_EQ(params.z, adjoint_source.get_z());

    // Get the computed force vector
    auto computed_force_vector = adjoint_source.get_force_vector();

    // Get the force vector from the solution
    auto expected_force_vector = solution.force_vector;

    // Check if the computed force vector matches the expected one
    EXPECT_EQ(computed_force_vector.extent(0), expected_force_vector.extent(0));

    // Compare each component of the force vector
    for (size_t i = 0; i < expected_force_vector.size(); ++i) {
      EXPECT_NEAR(computed_force_vector(i), expected_force_vector(i), 1e-5)
          << "Mismatch at index " << i;
    }
  }
}
