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
    specfem::sources::force<specfem::dimension::type::dim2> > {
  source_parameters() : x(0.0), z(0.0), angle(0.0) {};
  source_parameters(std::string name, type_real x, type_real z, type_real angle,
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

template <>
struct source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::force<specfem::dimension::type::dim2> > {
public:
  source_solution(type_real x, type_real y, type_real angle,
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

// Defining short hands for the source parameters and solution types
using ForceSource2DSolution =
    source_solution<specfem::dimension::type::dim2,
                    specfem::sources::force<specfem::dimension::type::dim2> >;
using ForceSource2DParameters =
    source_parameters<specfem::dimension::type::dim2,
                      specfem::sources::force<specfem::dimension::type::dim2> >;

using ForceSource2DParametersAndSolution =
    std::tuple<ForceSource2DParameters, ForceSource2DSolution>;
// Vector of pairs of force parameters and corresponding vector force solutions
template <>
std::vector<ForceSource2DParametersAndSolution> get_parameters_and_solutions<
    specfem::dimension::type::dim2,
    specfem::sources::force<specfem::dimension::type::dim2> >() {
  type_real sqrt2over2 = std::sqrt(2.0) / 2.0;

  return std::vector<ForceSource2DParametersAndSolution>{
    // Test elastic PSV source at origin with zero angle
    std::make_tuple(
        ForceSource2DParameters("elastic_psv and zero angle", 0.0, 0.0, 0.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic_psv),
        ForceSource2DSolution(0.0, 0.0, 0.0,
                              std::vector<type_real>{ 0., 1.0 })),
    // Test elastic PSV source at origin with 45 angle
    std::make_tuple(
        ForceSource2DParameters("elastic_psv and 45 degree angle", 0.0, 0.0,
                                45.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic_psv),
        ForceSource2DSolution(
            0.0, 0.0, 45.0, std::vector<type_real>{ sqrt2over2, sqrt2over2 })),
    // Test elastic isotropic source at origin with 90 degree angle
    std::make_tuple(ForceSource2DParameters(
                        "elastic_isotropic and 90 angle", 3.0, 3.0, 90.0,
                        specfem::wavefield::simulation_field::forward,
                        specfem::element::medium_tag::elastic_psv),
                    ForceSource2DSolution(3.0, 3.0, 90.0,
                                          std::vector<type_real>{ 1.0, 0.0 })),
    // Test elastic SH source at origin with angle angle should not affect the
    // force vector
    std::make_tuple(
        ForceSource2DParameters("elastic_sh and zero angle", 1.0, 1.0, 45.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic_sh),
        ForceSource2DSolution(1.0, 1.0, 45.0, std::vector<type_real>{ 1.0 })),
    // Test acoustic source at origin with 45 degree angle, which should not
    // affect the force vector
    std::make_tuple(
        ForceSource2DParameters("acoustic and 45 angle", 2.0, 2.0, 45.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::acoustic),
        ForceSource2DSolution(2.0, 2.0, 45.0, std::vector<type_real>{ 1.0 })),
    // Test elastic_psv_t source at origin with 45 angle
    std::make_tuple(
        ForceSource2DParameters("elastic_psv_t and 45 angle", 4.0, 4.0, 45.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic_psv_t),
        ForceSource2DSolution(
            4.0, 4.0, 45.0,
            std::vector<type_real>{ sqrt2over2, sqrt2over2, 0.0 }))
  };
}

// Factory function specialization for 2D Force Source
template <>
specfem::sources::force<specfem::dimension::type::dim2>
create_source<specfem::dimension::type::dim2,
              specfem::sources::force<specfem::dimension::type::dim2> >(
    const source_parameters<
        specfem::dimension::type::dim2,
        specfem::sources::force<specfem::dimension::type::dim2> > &parameters) {
  return specfem::sources::force<specfem::dimension::type::dim2>(
      parameters.x, parameters.z, parameters.angle,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}
