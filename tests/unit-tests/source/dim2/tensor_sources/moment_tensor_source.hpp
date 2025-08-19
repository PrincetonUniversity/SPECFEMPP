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
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> > {
  source_parameters() : x(0.0), z(0.0), Mxx(0.0), Mzz(0.0), Mxz(0.0) {};
  source_parameters(std::string name, type_real x, type_real z, type_real Mxx,
                    type_real Mzz, type_real Mxz,
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

template <>
struct source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> > {
public:
  source_solution(type_real x, type_real z, type_real Mxx, type_real Mzz,
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

// Defining short hands for the source parameters and solution types
using MomentTensorSource2DSolution = source_solution<
    specfem::dimension::type::dim2,
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> >;
using MomentTensorSource2DParameters = source_parameters<
    specfem::dimension::type::dim2,
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> >;

using MomentTensorSource2DParametersAndSolution =
    std::tuple<MomentTensorSource2DParameters, MomentTensorSource2DSolution>;
// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
template <>
std::vector<MomentTensorSource2DParametersAndSolution>
get_parameters_and_solutions<
    specfem::dimension::type::dim2,
    specfem::sources::moment_tensor<specfem::dimension::type::dim2> >() {
  return std::vector<MomentTensorSource2DParametersAndSolution>{
    // Test elastic P-SV moment tensor source

    std::make_tuple(
        MomentTensorSource2DParameters(
            "elastic_psv", 0.0, 0.0, 1.0, 2.0, 0.5,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic_psv),
        MomentTensorSource2DSolution(0.0, 0.0, 1.0, 2.0, 0.5,
                                     std::vector<std::vector<type_real> >{
                                         { 1.0, 0.5 }, { 0.5, 2.0 } })),
    // Test poroelastic moment tensor source (elastic tensor repeated twice)
    std::make_tuple(
        MomentTensorSource2DParameters(
            "poroelastic", 1.0, 1.0, 2.0, 3.0, 1.5,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::poroelastic),
        MomentTensorSource2DSolution(
            1.0, 1.0, 2.0, 3.0, 1.5,
            std::vector<std::vector<type_real> >{
                { 2.0, 1.5 }, { 1.5, 3.0 }, { 2.0, 1.5 }, { 1.5, 3.0 } })),
    // Test elastic P-SV-T moment tensor source (third component zero)
    std::make_tuple(MomentTensorSource2DParameters(
                        "elastic_psv_t", 2.0, 2.0, 0.8, 1.2, 0.3,
                        specfem::wavefield::simulation_field::forward,
                        specfem::element::medium_tag::elastic_psv_t),
                    MomentTensorSource2DSolution(
                        2.0, 2.0, 0.8, 1.2, 0.3,
                        std::vector<std::vector<type_real> >{
                            { 0.8, 0.3 }, { 0.3, 1.2 }, { 0.0, 0.0 } })),
    // Test electromagnetic TE moment tensor source
    std::make_tuple(
        MomentTensorSource2DParameters(
            "electromagnetic_te", 3.0, 3.0, 1.5, 2.5, 0.8,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::electromagnetic_te),
        MomentTensorSource2DSolution(
            3.0, 3.0, 1.5, 2.5, 0.8,
            std::vector<std::vector<type_real> >{ { 1.5, 0.8 }, { 0.8, 2.5 } }))
  };
}

// Factory function specialization for 2D Moment Tensor Source
template <>
specfem::sources::moment_tensor<specfem::dimension::type::dim2>
create_source<specfem::dimension::type::dim2,
              specfem::sources::moment_tensor<specfem::dimension::type::dim2> >(
    const source_parameters<
        specfem::dimension::type::dim2,
        specfem::sources::moment_tensor<specfem::dimension::type::dim2> >
        &parameters) {
  return specfem::sources::moment_tensor<specfem::dimension::type::dim2>(
      parameters.x, parameters.z, parameters.Mxx, parameters.Mzz,
      parameters.Mxz,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}

// Explicit template instantiations
