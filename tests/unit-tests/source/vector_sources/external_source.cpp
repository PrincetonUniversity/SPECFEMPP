#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct external_parameters {
  external_parameters() : x(0.0), z(0.0) {};
  external_parameters(std::string name, type_real x, type_real z,
                      specfem::wavefield::simulation_field wavefield_type,
                      specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), wavefield_type(wavefield_type),
        medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

struct external_solution {
public:
  external_solution(type_real x, type_real z,
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

// Vector of pairs of external parameters and corresponding vector force
// solutions
std::vector<std::tuple<external_parameters, external_solution> >
get_external_parameters_and_solutions() {
  return std::vector<std::tuple<external_parameters, external_solution> >{
    // Test acoustic external source
    std::make_tuple(
        external_parameters("acoustic", 0.0, 0.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::acoustic),
        external_solution(0.0, 0.0, std::vector<type_real>{ 1.0 })),
    // Test elastic SH external source
    std::make_tuple(
        external_parameters("elastic_sh", 1.0, 1.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::elastic_sh),
        external_solution(1.0, 1.0, std::vector<type_real>{ 1.0 })),
    // Test elastic P-SV external source
    std::make_tuple(
        external_parameters("elastic_psv", 2.0, 2.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::elastic_psv),
        external_solution(2.0, 2.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test poroelastic external source
    std::make_tuple(
        external_parameters("poroelastic", 3.0, 3.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::poroelastic),
        external_solution(3.0, 3.0,
                          std::vector<type_real>{ 1.0, 1.0, 1.0, 1.0 })),
    // Test electromagnetic TE external source
    std::make_tuple(
        external_parameters("electromagnetic_te", 4.0, 4.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::electromagnetic_te),
        external_solution(4.0, 4.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test elastic P-SV-T external source (note: external uses 1.0 for all
    // components)
    std::make_tuple(
        external_parameters("elastic_psv_t", 5.0, 5.0,
                            specfem::wavefield::simulation_field::forward,
                            specfem::element::medium_tag::elastic_psv_t),
        external_solution(5.0, 5.0, std::vector<type_real>{ 1.0, 1.0, 1.0 }))
  };
}

// Now for each parameter set, we will create an external source and compute the
// force vector using the get_force_vector method.

TEST(SOURCES, external_source_vector) {
  // Get the parameters and solutions
  auto parameters_and_solutions = get_external_parameters_and_solutions();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &params = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);

    // Add to trace for easier debugging
    SCOPED_TRACE("Testing external source for: " + params.name);

    // Create an external source
    specfem::sources::external external_source(
        params.x, params.z,
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        params.wavefield_type);

    // Set the medium tag
    external_source.set_medium_tag(params.medium_tag);

    // Compare x and z (this is just reassignment, no computation)
    EXPECT_REAL_EQ(params.x, external_source.get_x());
    EXPECT_REAL_EQ(params.z, external_source.get_z());

    // Get the computed force vector
    auto computed_force_vector = external_source.get_force_vector();

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
