
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
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> > {
  source_parameters() : x(0.0), z(0.0) {};
  source_parameters(std::string name, type_real x, type_real z,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), z(z), medium_tag(medium_tag) {};

  std::string name;                        ///< Name of the source
  type_real x;                             ///< x-coordinate of the source
  type_real z;                             ///< z-coordinate of the source
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> > {
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
using AdjointSource2DSolution = source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> >;
using AdjointSource2DParameters = source_parameters<
    specfem::dimension::type::dim2,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> >;

using AdjointSource2DParametersAndSolution =
    std::tuple<AdjointSource2DParameters, AdjointSource2DSolution>;
// Vector of pairs of adjoint parameters and corresponding vector force
// solutions
template <>
std::vector<AdjointSource2DParametersAndSolution> get_parameters_and_solutions<
    specfem::dimension::type::dim2,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> >() {
  return std::vector<AdjointSource2DParametersAndSolution>{
    // Test acoustic adjoint source
    std::make_tuple(
        AdjointSource2DParameters("acoustic", 0.0, 0.0,
                                  specfem::element::medium_tag::acoustic),
        AdjointSource2DSolution(0.0, 0.0, std::vector<type_real>{ 1.0 })),
    // Test elastic SH adjoint source
    std::make_tuple(
        AdjointSource2DParameters("elastic_sh", 1.0, 1.0,
                                  specfem::element::medium_tag::elastic_sh),
        AdjointSource2DSolution(1.0, 1.0, std::vector<type_real>{ 1.0 })),
    // Test elastic P-SV adjoint source
    std::make_tuple(
        AdjointSource2DParameters("elastic_psv", 2.0, 2.0,
                                  specfem::element::medium_tag::elastic_psv),
        AdjointSource2DSolution(2.0, 2.0, std::vector<type_real>{ 1.0, 1.0 })),
    // Test poroelastic adjoint source
    std::make_tuple(
        AdjointSource2DParameters("poroelastic", 3.0, 3.0,
                                  specfem::element::medium_tag::poroelastic),
        AdjointSource2DSolution(3.0, 3.0,
                                std::vector<type_real>{ 1.0, 1.0, 1.0, 1.0 })),
    // Test elastic P-SV-T adjoint source (third component should be 0)
    std::make_tuple(
        AdjointSource2DParameters("elastic_psv_t", 4.0, 4.0,
                                  specfem::element::medium_tag::elastic_psv_t),
        AdjointSource2DSolution(4.0, 4.0,
                                std::vector<type_real>{ 1.0, 1.0, 0.0 })),
    // Test electromagnetic TE adjoint source
    std::make_tuple(
        AdjointSource2DParameters(
            "electromagnetic_te", 5.0, 5.0,
            specfem::element::medium_tag::electromagnetic_te),
        AdjointSource2DSolution(5.0, 5.0, std::vector<type_real>{ 1.0, 1.0 }))
  };
}

// Factory function specialization for 2D Adjoint Source
template <>
specfem::sources::adjoint_source<specfem::dimension::type::dim2> create_source<
    specfem::dimension::type::dim2,
    specfem::sources::adjoint_source<specfem::dimension::type::dim2> >(
    const source_parameters<
        specfem::dimension::type::dim2,
        specfem::sources::adjoint_source<specfem::dimension::type::dim2> >
        &parameters) {
  return specfem::sources::adjoint_source<specfem::dimension::type::dim2>(
      parameters.x, parameters.z,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      "STA", "NET");
}

// Explicit template instantiations
