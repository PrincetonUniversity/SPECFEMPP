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
    specfem::dimension::type::dim3,
    specfem::sources::force<specfem::dimension::type::dim3> > {
  source_parameters() : x(0.0), y(0.0), z(0.0), fx(0.0), fy(0.0), fz(0.0) {};
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real fx, type_real fy, type_real fz,
                    specfem::wavefield::simulation_field wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), fx(fx), fy(fy), fz(fz),
        wavefield_type(wavefield_type), medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real y;      ///< y-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real fx;     ///< x-component of force
  type_real fy;     ///< y-component of force
  type_real fz;     ///< z-component of force
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::dimension::type::dim3,
    specfem::sources::force<specfem::dimension::type::dim3> > {
public:
  source_solution(type_real x, type_real y, type_real z,
                  std::vector<type_real> force_vector)
      : x(x), y(y), z(z) {
    this->force_vector = specfem::kokkos::HostView1d<type_real>(
        "force_vector", force_vector.size());
    for (size_t i = 0; i < force_vector.size(); ++i) {
      this->force_vector(i) = force_vector[i];
    }
  }

  type_real x; ///< x-coordinate of the source
  type_real y; ///< y-coordinate of the source
  type_real z; ///< z-coordinate of the source
  specfem::kokkos::HostView1d<type_real> force_vector; ///< Force vector in
                                                       ///< Kokkos format
};

// Vector of pairs of force parameters and corresponding vector force solutions
// Defining short hands for the source parameters and solution types
using ForceSource3DSolution =
    source_solution<specfem::dimension::type::dim3,
                    specfem::sources::force<specfem::dimension::type::dim3> >;
using ForceSource3DParameters =
    source_parameters<specfem::dimension::type::dim3,
                      specfem::sources::force<specfem::dimension::type::dim3> >;

using ForceSource3DParametersAndSolution =
    std::tuple<ForceSource3DParameters, ForceSource3DSolution>;
// Vector of pairs of force parameters and corresponding vector force solutions
template <>
std::vector<ForceSource3DParametersAndSolution> get_parameters_and_solutions<
    specfem::dimension::type::dim3,
    specfem::sources::force<specfem::dimension::type::dim3> >() {
  return std::vector<ForceSource3DParametersAndSolution>{
    // Test 3D elastic source with force in x-direction
    std::make_tuple(
        ForceSource3DParameters("3D elastic x-force", 0.0, 0.0, 0.0, 1.0, 0.0,
                                0.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic),
        ForceSource3DSolution(0.0, 0.0, 0.0,
                              std::vector<type_real>{ 1.0, 0.0, 0.0 })),
    // Test 3D elastic source with force in y-direction
    std::make_tuple(
        ForceSource3DParameters("3D elastic y-force", 1.0, 1.0, 1.0, 0.0, 1.0,
                                0.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic),
        ForceSource3DSolution(1.0, 1.0, 1.0,
                              std::vector<type_real>{ 0.0, 1.0, 0.0 })),
    // Test 3D elastic source with force in z-direction
    std::make_tuple(
        ForceSource3DParameters("3D elastic z-force", 2.0, 2.0, 2.0, 0.0, 0.0,
                                1.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic),
        ForceSource3DSolution(2.0, 2.0, 2.0,
                              std::vector<type_real>{ 0.0, 0.0, 1.0 })),
    // Test 3D elastic source with combined force
    std::make_tuple(
        ForceSource3DParameters("3D elastic combined force", 3.0, 3.0, 3.0, 0.5,
                                0.5, 0.5,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::elastic),
        ForceSource3DSolution(3.0, 3.0, 3.0,
                              std::vector<type_real>{ 0.5, 0.5, 0.5 })),
    // Test 3D acoustic source (single component)
    std::make_tuple(
        ForceSource3DParameters("3D acoustic", 4.0, 4.0, 4.0, 1.0, 0.0, 0.0,
                                specfem::wavefield::simulation_field::forward,
                                specfem::element::medium_tag::acoustic),
        ForceSource3DSolution(4.0, 4.0, 4.0, std::vector<type_real>{ 1.0 }))
  };
}

// Factory function specialization for 3D Force Source
template <>
specfem::sources::force<specfem::dimension::type::dim3>
create_source<specfem::dimension::type::dim3,
              specfem::sources::force<specfem::dimension::type::dim3> >(
    const source_parameters<
        specfem::dimension::type::dim3,
        specfem::sources::force<specfem::dimension::type::dim3> > &parameters) {
  return specfem::sources::force<specfem::dimension::type::dim3>(
      parameters.x, parameters.y, parameters.z, parameters.fx, parameters.fy,
      parameters.fz,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}

// Explicit template instantiations
