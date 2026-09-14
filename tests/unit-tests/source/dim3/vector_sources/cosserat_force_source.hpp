#include "../../source.hpp"

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

template <>
struct source_parameters<
    specfem::element::dimension_tag::dim3,
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3> > {
  source_parameters()
      : x(0.0), y(0.0), z(0.0), fx(0.0), fy(0.0), fz(0.0), fc_x(0.0), fc_y(0.0),
        fc_z(0.0) {};
  source_parameters(std::string name, type_real x, type_real y, type_real z,
                    type_real fx, type_real fy, type_real fz, type_real fc_x,
                    type_real fc_y, type_real fc_z,
                    specfem::simulation::field_type wavefield_type,
                    specfem::element::medium_tag medium_tag)
      : name(name), x(x), y(y), z(z), fx(fx), fy(fy), fz(fz), fc_x(fc_x),
        fc_y(fc_y), fc_z(fc_z), wavefield_type(wavefield_type),
        medium_tag(medium_tag) {};

  std::string name; ///< Name of the source
  type_real x;      ///< x-coordinate of the source
  type_real y;      ///< y-coordinate of the source
  type_real z;      ///< z-coordinate of the source
  type_real fx;     ///< x-component of the elastic force
  type_real fy;     ///< y-component of the elastic force
  type_real fz;     ///< z-component of the elastic force
  type_real fc_x;   ///< x-component of the rotational force
  type_real fc_y;   ///< y-component of the rotational force
  type_real fc_z;   ///< z-component of the rotational force
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield
  specfem::element::medium_tag medium_tag;        ///< Medium tag of the source
};

template <>
struct source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3> > {
public:
  source_solution(type_real x, type_real y, type_real z,
                  std::vector<type_real> force_vector)
      : x(x), y(y), z(z) {
    this->force_vector =
        Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>(
            "force_vector", force_vector.size());
    for (size_t i = 0; i < force_vector.size(); ++i) {
      this->force_vector(i) = force_vector[i];
    }
  }

  type_real x; ///< x-coordinate of the source
  type_real y; ///< y-coordinate of the source
  type_real z; ///< z-coordinate of the source
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
      force_vector; ///< Force vector in Kokkos format
};

// Defining short hands for the source parameters and solution types
using CosseratForceSource3DSolution = source_solution<
    specfem::element::dimension_tag::dim3,
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3> >;
using CosseratForceSource3DParameters = source_parameters<
    specfem::element::dimension_tag::dim3,
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3> >;

using CosseratForceSource3DParametersAndSolution =
    std::tuple<CosseratForceSource3DParameters, CosseratForceSource3DSolution>;
// Vector of pairs of cosserat force parameters and corresponding vector force
// solutions
template <>
std::vector<CosseratForceSource3DParametersAndSolution>
get_parameters_and_solutions<specfem::element::dimension_tag::dim3,
                             specfem::sources::cosserat_force<
                                 specfem::element::dimension_tag::dim3> >() {
  type_real sqrt2over2 = std::sqrt(2.0) / 2.0;

  return std::vector<CosseratForceSource3DParametersAndSolution>{
    // Test cosserat elastic source with pure x-direction force and no
    // rotational moment
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat x displacement force", 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            0.0, 0.0, 0.0,
            std::vector<type_real>{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 })),
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat y displacement force", 1.0, 1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            1.0, 1.0, 1.0,
            std::vector<type_real>{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 })),
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat z displacement force", 2.0, 2.0, 2.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            2.0, 2.0, 2.0,
            std::vector<type_real>{ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 })),
    // Rotation force tests
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat x rotational force", 3.0, 3.0, 3.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            3.0, 3.0, 3.0,
            std::vector<type_real>{ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 })),
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat y rotational force", 4.0, 4.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            4.0, 4.0, 4.0,
            std::vector<type_real>{ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 })),
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat z rotational force", 5.0, 5.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            5.0, 5.0, 5.0,
            std::vector<type_real>{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 })),
    // all displacement forces
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat combined displacement force", 6.0, 6.0, 6.0,
            0.5, 0.5, 0.5, 0.0, 0.0, 0.0,
            specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            6.0, 6.0, 6.0,
            std::vector<type_real>{ 0.5, 0.5, 0.5, 0.0, 0.0, 0.0 })),
    // all rotational forces
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat combined rotational force", 7.0, 7.0, 7.0, 0.0,
            0.0, 0.0, 0.5, 0.5, 0.5, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            7.0, 7.0, 7.0,
            std::vector<type_real>{ 0.0, 0.0, 0.0, 0.5, 0.5, 0.5 })),
    // all components combined
    std::make_tuple(
        CosseratForceSource3DParameters(
            "3D Elastic Cosserat combined force and moment", 8.0, 8.0, 8.0, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, specfem::simulation::field_type::forward,
            specfem::element::medium_tag::elastic_spin),
        CosseratForceSource3DSolution(
            8.0, 8.0, 8.0,
            std::vector<type_real>{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 })),
  };
}

// Factory function specialization for 3D Cosserat Force Source
template <>
specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3>
create_source<
    specfem::element::dimension_tag::dim3,
    specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3> >(
    const source_parameters<specfem::element::dimension_tag::dim3,
                            specfem::sources::cosserat_force<
                                specfem::element::dimension_tag::dim3> >
        &parameters) {
  return specfem::sources::cosserat_force<
      specfem::element::dimension_tag::dim3>(
      parameters.x, parameters.y, parameters.z, parameters.fx, parameters.fy,
      parameters.fz, parameters.fc_x, parameters.fc_y, parameters.fc_z,
      std::make_unique<specfem::source_time_functions::Ricker>(10, 0.01, 1.0,
                                                               0.0, 1.0, false),
      parameters.wavefield_type);
}
