#pragma once

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Collocated force source
 *
 */
class cosserat_force : public source {

public:
  /**
   * @brief Default source constructor
   *
   */
  cosserat_force() {};
  /**
   * @brief Construct a new collocated force object
   *
   * @param cosserat_source A YAML node defining cosserat force source
   * @param dt Time increment in the simulation. Used to calculate dominant
   * frequecy of Dirac source.
   */
  cosserat_force(YAML::Node &Node, const int nsteps, const type_real dt,
                 const specfem::wavefield::simulation_field wavefield_type)
      : angle([](YAML::Node &Node) -> type_real {
          if (Node["angle"]) {
            return Node["angle"].as<type_real>();
          } else {
            return 0.0;
          }
        }(Node)),
        f(Node["f"].as<type_real>()), fc(Node["fc"].as<type_real>()),
        wavefield_type(wavefield_type),
        specfem::sources::source(Node, nsteps, dt) {};

  type_real get_angle() const { return angle; }
  type_real get_f() const { return f; }
  type_real get_fc() const { return fc; }
  /**
   * @brief Construct a new collocated force object
   */
  cosserat_force(
      type_real x, type_real z, type_real f, type_real fc, type_real angle,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : f(f), fc(fc), angle(angle), wavefield_type(wavefield_type),
        specfem::sources::source(x, z, std::move(forcing_function)) {};
  /**
   * @brief User output
   *
   */
  std::string print() const override;

  void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
      specfem::kokkos::HostView3d<type_real> source_array) override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(const specfem::sources::source &other) const override;
  bool operator!=(const specfem::sources::source &other) const override;

private:
  type_real angle; ///< Angle of the elastic force source
  type_real f;     ///< Factor to scale the elastic force
  type_real fc;    ///< Factor to scale the rotational force
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
};

} // namespace sources
} // namespace specfem
