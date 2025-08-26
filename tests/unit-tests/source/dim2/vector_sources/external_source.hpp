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
    specfem::sources::external<specfem::dimension::type::dim2> > {
  source_parameters() : x(0.0), z(0.0) {};
  source_parameters(std::string name, type_real x, type_real z,
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

template <>
struct source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::external<specfem::dimension::type::dim2> > {
public:
  source_solution(type_real x, type_real z, std::vector<type_real> force_vector)
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

// Defining short hands for the source parameters and solution types
using ExternalSource2DSolution = source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::external<specfem::dimension::type::dim2> >;
using ExternalSource2DParameters = source_parameters<
    specfem::dimension::type::dim2,
    specfem::sources::external<specfem::dimension::type::dim2> >;

using ExternalSource2DParametersAndSolution =
    std::tuple<ExternalSource2DParameters, ExternalSource2DSolution>;
// Vector of pairs of external parameters and corresponding vector force
// solutions
template <>
std::vector<ExternalSource2DParametersAndSolution> get_parameters_and_solutions<
    specfem::dimension::type::dim2,
    specfem::sources::external<specfem::dimension::type::dim2> >() {
  return std::vector<ExternalSource2DParametersAndSolution>{
    // Test acoustic external source
    std::make_tuple(
        ExternalSource2DParameters(
            "acoustic", 0.0, 0.0, specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::acoustic),
        ExternalSource2DSolution(0.0, 0.0, std::vector<type_real>{ 1.0 })),
    // Test elastic SH external source
    std::make_tuple(
        ExternalSource2DParameters(
            "elastic_sh", 1.0, 1.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_sh),
        ExternalSource2DSolution(1.0, 1.0, std::vector<type_real>{ 1.0 })),
    // Test elastic P-SV external source
    std::make_tuple(
        ExternalSource2DParameters(
            "elastic_psv", 2.0, 2.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv),
        ExternalSource2DSolution(2.0, 2.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test poroelastic external source
    std::make_tuple(
        ExternalSource2DParameters(
            "poroelastic", 3.0, 3.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::poroelastic),
        ExternalSource2DSolution(3.0, 3.0,
                                 std::vector<type_real>{ 1.0, 1.0, 1.0, 1.0 })),
    // Test electromagnetic TE external source
    std::make_tuple(
        ExternalSource2DParameters(
            "electromagnetic_te", 4.0, 4.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::electromagnetic_te),
        ExternalSource2DSolution(4.0, 4.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test elastic P-SV-T external source (note: external uses 1.0 for all
    // components)
    std::make_tuple(ExternalSource2DParameters(
                        "elastic_psv_t", 5.0, 5.0,
                        specfem::wavefield::simulation_field::forward,
                        specfem::element::medium_tag::elastic_psv_t),
                    ExternalSource2DSolution(
                        5.0, 5.0, std::vector<type_real>{ 1.0, 1.0, 1.0 }))
  };
}

// Factory function specialization for 2D External Source
template <>
specfem::sources::external<specfem::dimension::type::dim2>
create_source<specfem::dimension::type::dim2,
              specfem::sources::external<specfem::dimension::type::dim2> >(
    const source_parameters<
        specfem::dimension::type::dim2,
        specfem::sources::external<specfem::dimension::type::dim2> >
        &parameters) {
  return specfem::sources::external<specfem::dimension::type::dim2>(
      parameters.x, parameters.z,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}
