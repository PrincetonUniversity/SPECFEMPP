#pragma once

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"

#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Collocated force source
 *
 * This class implements a collocated force source in 2D that applies a
 * directional force at a specific location in the simulation domain.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     10,    // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 2D force source at (2.5, 3.0) with 45-degree angle
 * auto force_source = specfem::sources::force<specfem::dimension::type::dim2>(
 *     2.5,  // x-coordinate
 *     3.0,  // z-coordinate
 *     45.0, // angle in degrees
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium type
 * force_source.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Get the force vector (depends on medium type)
 * auto force_vector = force_source.get_force_vector();
 * @endcode
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
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(
      type_real x, type_real z, type_real angle,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : angle(angle), wavefield_type(wavefield_type),
        vector_source(x, z, std::move(source_time_function)) {};

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
   * Returns the 2D force vector for this directional force source:
   *
   * \f[
   * \mathbf{f}_{2D} = \begin{cases}
   * [\sin(\theta), -\cos(\theta)] & \text{elastic PSV: x,z components} \\
   * [1.0] & \text{elastic SH: out-of-plane component} \\
   * [1.0] & \text{acoustic: pressure amplitude} \\
   * [\sin(\theta), -\cos(\theta), \sin(\theta), -\cos(\theta)] &
   * \text{poroelastic} \\
   * [\sin(\theta), -\cos(\theta), 0.0] & \text{elastic PSV-T}
   * \end{cases}
   * \f]
   *
   * Where:
   * - \f$f\f$ is the force magnitude (normalized to 1.0)
   * - \f$\theta\f$ is the force angle in degrees from horizontal
   * - Components depend on the medium and wave field type
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector with size depending on medium type
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
  static constexpr const char *name = "2-D force";

private:
  type_real angle; ///< Angle of force source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
