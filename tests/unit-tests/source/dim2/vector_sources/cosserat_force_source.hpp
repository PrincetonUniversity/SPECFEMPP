#include "../../source.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

template <>
struct source_parameters<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> > {
  source_parameters() : x(0.0), z(0.0), f(0.0), fc(0.0), angle(0.0) {};
  source_parameters(std::string name, type_real x, type_real z, type_real f,
                    type_real fc, type_real angle,
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

template <>
struct source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> > {
public:
  source_solution(type_real x, type_real z, type_real f, type_real fc,
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

// Defining short hands for the source parameters and solution types
using CosseratForceSource2DSolution = source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> >;
using CosseratForceSource2DParameters = source_parameters<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> >;

using CosseratForceSource2DParametersAndSolution =
    std::tuple<CosseratForceSource2DParameters, CosseratForceSource2DSolution>;
// Vector of pairs of cosserat force parameters and corresponding vector force
// solutions
template <>
std::vector<CosseratForceSource2DParametersAndSolution>
get_parameters_and_solutions<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> >() {
  type_real sqrt2over2 = std::sqrt(2.0) / 2.0;

  return std::vector<CosseratForceSource2DParametersAndSolution>{
    // Test cosserat elastic PSV-T source at origin with zero angle
    std::make_tuple(
        CosseratForceSource2DParameters(
            "cosserat_elastic_psv_t and zero angle", 0.0, 0.0, 1.0, 2.0, 0.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv_t),
        CosseratForceSource2DSolution(
            0.0, 0.0, 1.0, 2.0, 0.0, std::vector<type_real>{ 0.0, -1.0, 2.0 })),
    // Test cosserat elastic PSV-T source with 45 degree angle
    std::make_tuple(
        CosseratForceSource2DParameters(
            "cosserat_elastic_psv_t and 45 angle", 4.0, 4.0, 2.0, 1.5, 45.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv_t),
        CosseratForceSource2DSolution(
            4.0, 4.0, 2.0, 1.5, 45.0,
            std::vector<type_real>{ static_cast<type_real>(2.0) * sqrt2over2,
                                    static_cast<type_real>(-2.0) * sqrt2over2,
                                    static_cast<type_real>(1.5) })),
    // Test cosserat elastic PSV-T source with 90 degree angle
    std::make_tuple(
        CosseratForceSource2DParameters(
            "cosserat_elastic_psv_t and 90 angle", 3.0, 3.0, 1.5, 0.5, 90.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv_t),
        CosseratForceSource2DSolution(3.0, 3.0, 1.5, 0.5, 90.0,
                                      std::vector<type_real>{ 1.5, 0.0, 0.5 })),
    // Test cosserat elastic PSV-T source with 180 degree angle
    std::make_tuple(
        CosseratForceSource2DParameters(
            "cosserat_elastic_psv_t and 180 angle", 2.0, 2.0, 0.8, 1.2, 180.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv_t),
        CosseratForceSource2DSolution(2.0, 2.0, 0.8, 1.2, 180.0,
                                      std::vector<type_real>{ 0.0, 0.8, 1.2 }))
  };
}

// Factory function specialization for 2D Cosserat Force Source
template <>
specfem::sources::cosserat_force<specfem::dimension::type::dim2> create_source<
    specfem::dimension::type::dim2,
    specfem::sources::cosserat_force<specfem::dimension::type::dim2> >(
    const source_parameters<
        specfem::dimension::type::dim2,
        specfem::sources::cosserat_force<specfem::dimension::type::dim2> >
        &parameters) {
  return specfem::sources::cosserat_force<specfem::dimension::type::dim2>(
      parameters.x, parameters.z, parameters.f, parameters.fc, parameters.angle,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}
