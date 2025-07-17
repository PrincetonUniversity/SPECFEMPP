#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct force_parameters {
  force_parameters() : x(0.0), z(0.0), angle(0.0) {};
  force_parameters(std::string name, type_real x, type_real z, type_real angle,
                   specfem::wavefield::simulation_field wavefield_type,
                   specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), angle(angle), wavefield_type(wavefield_type),
        medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real angle;  ///< angle of the force source in degrees
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

struct force_solution {
public:
  force_solution(type_real x, type_real y, type_real angle,
                 std::vector<type_real> force_vector)
      : x(x), y(y), angle(angle) {
    this->force_vector = specfem::kokkos::HostView1d<type_real>(
        "force_vector", force_vector.size());
    for (size_t i = 0; i < force_vector.size(); ++i) {
      this->force_vector(i) = force_vector[i];
    }
  }

  type_real x;     ///< x-component of the force vector
  type_real y;     ///< y-component of the force vector
  type_real angle; ///< angle of the force vector in degrees
  specfem::kokkos::HostView1d<type_real> force_vector; ///< Force vector in
                                                       ///< Kokkos format
};

type_real sqrt2over2 = std::sqrt(2.0) / 2.0;

// Vector of pairs of force parameters and corresponding vector force solutions
std::vector<std::tuple<force_parameters, force_solution> >
get_parameters_and_solutions() {
  return std::vector<std::tuple<force_parameters, force_solution> >{
    // Test elastic PSV source at origin with zero angle
    std::make_tuple(
        force_parameters("elastic_psv and zero angle", 0.0, 0.0, 0.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::elastic_psv),
        force_solution(0.0, 0.0, 0.0, std::vector<type_real>{ 0., -1.0 })),
    // Test elastic_psv_t source at origin with 45 angle
    std::make_tuple(
        force_parameters("elastic_psv_t and 45 angle", 4.0, 4.0, 45.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::elastic_psv_t),
        force_solution(4.0, 4.0, 45.0,
                       std::vector<type_real>{ sqrt2over2, -sqrt2over2, 0.0 })),
    // Test elastic isotropic source at origin with 90 degree angle
    std::make_tuple(
        force_parameters("elastic_isotropic and 90 angle", 3.0, 3.0, 90.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::elastic_psv),
        force_solution(3.0, 3.0, 90.0, std::vector<type_real>{ 1.0, 0.0 })),
    // Test elastic SH source at origin with angle angle should not affect the
    // force vector
    std::make_tuple(
        force_parameters("elastic_sh and zero angle", 1.0, 1.0, 45.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::elastic_sh),
        force_solution(1.0, 1.0, 45.0, std::vector<type_real>{ 1.0 })),
    // Test acoustic source at origin with 45 degree angle, which should not
    // affect the force vector
    std::make_tuple(
        force_parameters("acoustic and 45 angle", 2.0, 2.0, 45.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::acoustic),
        force_solution(2.0, 2.0, 45.0, std::vector<type_real>{ 1.0 })),
    // Test elastic_psv_t source at origin with 45 angle
    std::make_tuple(
        force_parameters("elastic_psv_t and 45 angle", 4.0, 4.0, 45.0,
                         specfem::wavefield::simulation_field::forward,
                         specfem::element::medium_tag::elastic_psv_t),
        force_solution(4.0, 4.0, 45.0,
                       std::vector<type_real>{ sqrt2over2, -sqrt2over2, 0.0 }))
  };
}

// Now for each parameter set, we will create a force source and compute the
// force vector using the get_force_vector method.

TEST(SOURCES, force_source_vector) {
  // Get the parameters and solutions
  auto parameters_and_solutions = get_parameters_and_solutions();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &params = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);

    // Add to trace for easier debugging
    SCOPED_TRACE("Testing force source for: " + params.name);

    // Create a force source
    specfem::sources::force force_source(
        params.x, params.z, params.angle,
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        params.wavefield_type);

    // Set the medium tag
    force_source.set_medium_tag(params.medium_tag);

    // Compare x,z, and angle (this is just reassignement no computation)
    EXPECT_REAL_EQ(params.x, force_source.get_x());
    EXPECT_REAL_EQ(params.z, force_source.get_z());
    EXPECT_REAL_EQ(params.angle, force_source.get_angle());

    // Get the computed force vector
    auto computed_force_vector = force_source.get_force_vector();

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
