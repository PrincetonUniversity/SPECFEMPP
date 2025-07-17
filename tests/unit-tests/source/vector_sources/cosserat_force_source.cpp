#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

struct cosserat_force_parameters {
  cosserat_force_parameters() : x(0.0), z(0.0), f(0.0), fc(0.0), angle(0.0) {};
  cosserat_force_parameters(std::string name, type_real x, type_real z,
                            type_real f, type_real fc, type_real angle,
                            specfem::wavefield::simulation_field wavefield_type,
                            specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), f(f), fc(fc), angle(angle),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real f;      ///< Factor to scale the elastic force
  type_real fc;     ///< Factor to scale the rotational force
  type_real angle;  ///< angle of the force source in degrees
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

struct cosserat_force_solution {
public:
  cosserat_force_solution(type_real x, type_real z, type_real f, type_real fc,
                          type_real angle, std::vector<type_real> force_vector)
      : x(x), z(z), f(f), fc(fc), angle(angle) {
    this->force_vector = specfem::kokkos::HostView1d<type_real>(
        "force_vector", force_vector.size());
    for (size_t i = 0; i < force_vector.size(); ++i) {
      this->force_vector(i) = force_vector[i];
    }
  }

  type_real x;     ///< x-coordinate of the source
  type_real z;     ///< z-coordinate of the source
  type_real f;     ///< Factor to scale the elastic force
  type_real fc;    ///< Factor to scale the rotational force
  type_real angle; ///< angle of the force source in degrees
  specfem::kokkos::HostView1d<type_real> force_vector; ///< Force vector in
                                                       ///< Kokkos format
};

// Vector of pairs of cosserat force parameters and corresponding vector force
// solutions
std::vector<std::tuple<cosserat_force_parameters, cosserat_force_solution> >
get_cosserat_parameters_and_solutions() {
  type_real sqrt2over2 = std::sqrt(2.0) / 2.0;

  return std::vector<
      std::tuple<cosserat_force_parameters, cosserat_force_solution> >{
    // Test cosserat elastic PSV-T source at origin with zero angle
    std::make_tuple(
        cosserat_force_parameters("cosserat_elastic_psv_t and zero angle", 0.0,
                                  0.0, 1.0, 2.0, 0.0,
                                  specfem::wavefield::simulation_field::forward,
                                  specfem::element::medium_tag::elastic_psv_t),
        cosserat_force_solution(0.0, 0.0, 1.0, 2.0, 0.0,
                                std::vector<type_real>{ 0.0, -1.0, 2.0 })),
    // Test cosserat elastic PSV-T source with 45 degree angle
    std::make_tuple(
        cosserat_force_parameters("cosserat_elastic_psv_t and 45 angle", 4.0,
                                  4.0, 2.0, 1.5, 45.0,
                                  specfem::wavefield::simulation_field::forward,
                                  specfem::element::medium_tag::elastic_psv_t),
        cosserat_force_solution(
            4.0, 4.0, 2.0, 1.5, 45.0,
            std::vector<type_real>{ static_cast<type_real>(2.0) * sqrt2over2,
                                    static_cast<type_real>(-2.0) * sqrt2over2,
                                    static_cast<type_real>(1.5) })),
    // Test cosserat elastic PSV-T source with 90 degree angle
    std::make_tuple(
        cosserat_force_parameters("cosserat_elastic_psv_t and 90 angle", 3.0,
                                  3.0, 1.5, 0.5, 90.0,
                                  specfem::wavefield::simulation_field::forward,
                                  specfem::element::medium_tag::elastic_psv_t),
        cosserat_force_solution(3.0, 3.0, 1.5, 0.5, 90.0,
                                std::vector<type_real>{ 1.5, 0.0, 0.5 })),
    // Test cosserat elastic PSV-T source with 180 degree angle
    std::make_tuple(
        cosserat_force_parameters("cosserat_elastic_psv_t and 180 angle", 2.0,
                                  2.0, 0.8, 1.2, 180.0,
                                  specfem::wavefield::simulation_field::forward,
                                  specfem::element::medium_tag::elastic_psv_t),
        cosserat_force_solution(2.0, 2.0, 0.8, 1.2, 180.0,
                                std::vector<type_real>{ 0.0, 0.8, 1.2 }))
  };
}

// Now for each parameter set, we will create a cosserat force source and
// compute the force vector using the get_force_vector method.

TEST(SOURCES, cosserat_force_source_vector) {
  // Get the parameters and solutions
  auto parameters_and_solutions = get_cosserat_parameters_and_solutions();

  for (const auto &param_solution : parameters_and_solutions) {
    const auto &params = std::get<0>(param_solution);
    const auto &solution = std::get<1>(param_solution);

    // Add to trace for easier debugging
    SCOPED_TRACE("Testing cosserat force source for: " + params.name);

    // Create a cosserat force source
    specfem::sources::cosserat_force cosserat_force_source(
        params.x, params.z, params.f, params.fc, params.angle,
        std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                            1.0, false),
        params.wavefield_type);

    // Set the medium tag
    cosserat_force_source.set_medium_tag(params.medium_tag);

    // Compare x, z, f, fc, and angle (this is just reassignment, no
    // computation)
    EXPECT_REAL_EQ(params.x, cosserat_force_source.get_x());
    EXPECT_REAL_EQ(params.z, cosserat_force_source.get_z());
    EXPECT_REAL_EQ(params.f, cosserat_force_source.get_f());
    EXPECT_REAL_EQ(params.fc, cosserat_force_source.get_fc());
    EXPECT_REAL_EQ(params.angle, cosserat_force_source.get_angle());

    // Get the computed force vector
    auto computed_force_vector = cosserat_force_source.get_force_vector();

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
