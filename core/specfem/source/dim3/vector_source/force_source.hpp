#pragma once

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
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
class force<specfem::dimension::type::dim3>
    : public vector_source<specfem::dimension::type::dim3> {

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
      : vector_source(Node, nsteps, dt), fx(Node["fx"].as<type_real>()),
        fy(Node["fy"].as<type_real>()), fz(Node["fz"].as<type_real>()),
        wavefield_type(wavefield_type) {};

  /**
   * @brief Construct a new collocated force object
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(type_real x, type_real y, type_real z, type_real fx, type_real fy,
        type_real fz,
        std::unique_ptr<specfem::forcing_function::stf> forcing_function,
        const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type),
        vector_source(x, y, z, std::move(forcing_function)), fx(fx), fy(fy),
        fz(fz) {};

  /**
   * @brief User output
   *
   */
  std::string print() const override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  /**
   * @brief Get the forcing function
   *
   */
  bool operator==(const specfem::sources::source<specfem::dimension::type::dim3>
                      &other) const override;
  bool operator!=(const specfem::sources::source<specfem::dimension::type::dim3>
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

public:
  static constexpr const char *name = "3-D force";

private:
  type_real fx;                                        ///< Force in x-direction
  type_real fy;                                        ///< Force in y-direction
  type_real fz;                                        ///< Force in z-direction
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
