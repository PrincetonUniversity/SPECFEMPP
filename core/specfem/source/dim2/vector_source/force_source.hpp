#pragma once

#include "../vector_source.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
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
template <>
class force<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

public:
  /**
   * @brief Default source constructor
   *
   */
  force() {};
  /**
   * @brief Construct a new collocated force object
   *
   * @param force_source A YAML node defining force source
   * @param dt Time increment in the simulation. Used to calculate dominant
   * frequecy of Dirac source.
   */
  force(YAML::Node &Node, const int nsteps, const type_real dt,
        const specfem::wavefield::simulation_field wavefield_type)
      : angle([](YAML::Node &Node) -> type_real {
          if (Node["angle"]) {
            return Node["angle"].as<type_real>();
          } else {
            return 0.0;
          }
        }(Node)),
        wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {};

  /**
   * @brief Construct a new collocated force object
   *
   * @param x x-coordinate of source
   * @param y z-coordinate of source
   * @param angle angle of force source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(type_real x, type_real z, type_real angle,
        std::unique_ptr<specfem::forcing_function::stf> forcing_function,
        const specfem::wavefield::simulation_field wavefield_type)
      : angle(angle), wavefield_type(wavefield_type),
        vector_source(x, z, std::move(forcing_function)) {};

  /**
   * @brief User output
   *
   */
  std::string print() const override;

  /**
   * @brief Get the angle of the force source
   *
   * @return type_real angle of force source
   */
  type_real get_angle() const { return angle; }

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  /**
   * @brief Get the forcing function
   *
   */
  bool operator==(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;
  bool operator!=(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;

  /**
   * @brief Get the force vector
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  specfem::kokkos::HostView1d<type_real> get_force_vector() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  const static std::string name;
  type_real angle; ///< Angle of force source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
