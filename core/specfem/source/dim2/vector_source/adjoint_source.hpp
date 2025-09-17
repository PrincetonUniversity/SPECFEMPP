#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
/**
 * @brief Adjoint source
 *
 * This class implements an adjoint source in 2D, which is used in adjoint
 * simulations for seismic inversion and sensitivity analysis. Adjoint sources
 * are typically placed at receiver locations and represent data residuals.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function for the adjoint source
 * auto stf = std::make_unique<specfem::forcing_function::Ricker>(
 *     20.0,  // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 2D adjoint source at receiver location (12.5, 8.3)
 * auto adj_source =
 * specfem::sources::adjoint_source<specfem::dimension::type::dim2>( 12.5,  //
 * x-coordinate (receiver location) 8.3,   // z-coordinate (receiver location)
 *     std::move(stf),
 *     "STA01",    // station name
 *     "NETWORK"   // network name
 * );
 *
 * // Set the medium type where the adjoint source is located
 * adj_source.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Get the force vector (unit vector components based on medium)
 * auto force_vector = adj_source.get_force_vector();
 *
 * // Adjoint sources always return adjoint wavefield type
 * assert(adj_source.get_wavefield_type() ==
 *        specfem::wavefield::simulation_field::adjoint);
 * @endcode
 *
 */
template <>
class adjoint_source<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

public:
  adjoint_source() {};

  adjoint_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function,
      const std::string &station_name, const std::string &network_name)
      : vector_source(x, z, std::move(forcing_function)),
        station_name(station_name), network_name(network_name) {};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        vector_source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return specfem::wavefield::simulation_field::adjoint;
  }

  std::string print() const override;

  /**
   * @brief Get the force vector
   *
   * Returns a unit force vector for adjoint source computations:
   *
   * \f[
   * \mathbf{f}_{adjoint} = \begin{cases}
   * [1.0] & \text{acoustic: unit pressure amplitude} \\
   * [1.0] & \text{elastic SH: unit force out-of-plane} \\
   * [1.0, 1.0] & \text{elastic PSV: unit forces in x,z directions} \\
   * [1.0, 1.0, 1.0, 1.0] & \text{poroelastic: solid/fluid phases} \\
   * [1.0, 1.0] & \text{electromagnetic TE} \\
   * [1.0, 1.0, 0.0] & \text{elastic PSV-T: no rotation component}
   * \end{cases}
   * \f]
   *
   * Where the unit components provide the basis for adjoint computations
   * in full waveform inversion. The adjoint source acts as a time-reversed
   * receiver that backpropagates data residuals through the medium.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Unit force vector with size depending on medium type
   */
  specfem::kokkos::HostView1d<type_real> get_force_vector() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

  static constexpr const char *name = "2-D adjoint source";

private:
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
