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
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> > {
  source_parameters()
      : x(0.0), y(0.0), z(0.0), Mxx(0.0), Myy(0.0), Mzz(0.0), Mxy(0.0),
        Mxz(0.0), Myz(0.0) {};
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real Mxx, type_real Myy, type_real Mzz, type_real Mxy,
                    type_real Mxz, type_real Myz,
                    specfem::wavefield::simulation_field wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy),
        Mxz(Mxz), Myz(Myz), wavefield_type(wavefield_type),
        medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real y;      ///< y-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real Mxx;    ///< Mxx component of moment tensor
  type_real Myy;    ///< Myy component of moment tensor
  type_real Mzz;    ///< Mzz component of moment tensor
  type_real Mxy;    ///< Mxy component of moment tensor
  type_real Mxz;    ///< Mxz component of moment tensor
  type_real Myz;    ///< Myz component of moment tensor
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag; ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::dimension::type::dim3,
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> > {
public:
  source_solution(type_real x, type_real y, type_real z, type_real Mxx,
                  type_real Myy, type_real Mzz, type_real Mxy, type_real Mxz,
                  type_real Myz,
                  std::vector<std::vector<type_real> > source_tensor)
      : x(x), y(y), z(z), Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy), Mxz(Mxz),
        Myz(Myz) {
    this->source_tensor = specfem::kokkos::HostView2d<type_real>(
        "source_tensor", source_tensor.size(), source_tensor[0].size());
    for (size_t i = 0; i < source_tensor.size(); ++i) {
      for (size_t j = 0; j < source_tensor[i].size(); ++j) {
        this->source_tensor(i, j) = source_tensor[i][j];
      }
    }
  }

  type_real x;   ///< x-coordinate of the source
  type_real y;   ///< y-coordinate of the source
  type_real z;   ///< z-coordinate of the source
  type_real Mxx; ///< Mxx component of moment tensor
  type_real Myy; ///< Myy component of moment tensor
  type_real Mzz; ///< Mzz component of moment tensor
  type_real Mxy; ///< Mxy component of moment tensor
  type_real Mxz; ///< Mxz component of moment tensor
  type_real Myz; ///< Myz component of moment tensor
  specfem::kokkos::HostView2d<type_real> source_tensor; ///< Source tensor in
                                                        ///< Kokkos format
};

// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
// Defining short hands for the source parameters and solution types
using MomentTensorSource3DSolution = source_solution<
    specfem::dimension::type::dim3,
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> >;
using MomentTensorSource3DParameters = source_parameters<
    specfem::dimension::type::dim3,
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> >;

using MomentTensorSource3DParametersAndSolution =
    std::tuple<MomentTensorSource3DParameters, MomentTensorSource3DSolution>;
// Vector of pairs of moment tensor parameters and corresponding tensor
// solutions
template <>
std::vector<MomentTensorSource3DParametersAndSolution>
get_parameters_and_solutions<
    specfem::dimension::type::dim3,
    specfem::sources::moment_tensor<specfem::dimension::type::dim3> >() {
  return std::vector<MomentTensorSource3DParametersAndSolution>{
    // Test 3D elastic moment tensor source (simple diagonal)
    std::make_tuple(
        MomentTensorSource3DParameters(
            "3D elastic diagonal", 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic),
        MomentTensorSource3DSolution(
            0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0,
            std::vector<std::vector<type_real> >{
                { 1.0, 0.0, 0.0 }, { 0.0, 2.0, 0.0 }, { 0.0, 0.0, 3.0 } })),
    // Test 3D elastic moment tensor source (full tensor)
    std::make_tuple(
        MomentTensorSource3DParameters(
            "3D elastic full", 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            specfem::wavefield::simulation_field::forward,
            specfem::element::medium_tag::elastic),
        MomentTensorSource3DSolution(
            1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7,
            std::vector<std::vector<type_real> >{
                { 1.0, 0.5, 0.6 }, { 0.5, 2.0, 0.7 }, { 0.6, 0.7, 3.0 } })),
  };
}

// Factory function specialization for 3D Moment Tensor Source
template <>
specfem::sources::moment_tensor<specfem::dimension::type::dim3>
create_source<specfem::dimension::type::dim3,
              specfem::sources::moment_tensor<specfem::dimension::type::dim3> >(
    const source_parameters<
        specfem::dimension::type::dim3,
        specfem::sources::moment_tensor<specfem::dimension::type::dim3> >
        &parameters) {
  return specfem::sources::moment_tensor<specfem::dimension::type::dim3>(
      parameters.x, parameters.y, parameters.z, parameters.Mxx, parameters.Myy,
      parameters.Mzz, parameters.Mxy, parameters.Mxz, parameters.Myz,
      std::make_unique<specfem::forcing_function::Ricker>(10, 0.01, 1.0, 0.0,
                                                          1.0, false),
      parameters.wavefield_type);
}

// Explicit template instantiations
