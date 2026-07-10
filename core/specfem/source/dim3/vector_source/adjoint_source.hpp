#pragma once

#include "specfem/enums.hpp"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
/**
 * @brief Adjoint source in 3D
 *
 * This class implements an adjoint source in 3D, which is used in adjoint
 * simulations for seismic inversion and sensitivity analysis. Adjoint sources
 * are typically placed at receiver locations and represent data residuals.
 *
 * @par Usage Example
 * @code
 * // Create an external source time function for the adjoint source
 * auto stf = std::make_unique<specfem::source_time_functions::external>(...);
 *
 * // Create a 3D adjoint source at receiver location (67000, 22732, -50)
 * auto adj_source =
 *     specfem::sources::adjoint_source<specfem::element::dimension_tag::dim3>(
 *         67000.0,  // x-coordinate (receiver location)
 *         22732.0,  // y-coordinate (receiver location)
 *         -50.0,    // z-coordinate (receiver location)
 *         std::move(stf),
 *         "X20",    // station name
 *         "DB"      // network name
 *     );
 *
 * // Set the medium type where the adjoint source is located
 * adj_source.set_medium_tag(specfem::element::medium_tag::acoustic);
 *
 * // Get the force vector (1 component for acoustic, 3 for elastic)
 * auto force_vector = adj_source.get_force_vector();
 *
 * // Adjoint sources always return adjoint wavefield type
 * assert(adj_source.get_wavefield_type() ==
 *        specfem::simulation::field_type::adjoint);
 * @endcode
 *
 */
template <>
class adjoint_source<specfem::element::dimension_tag::dim3>
    : public vector_source<specfem::element::dimension_tag::dim3> {

public:
  adjoint_source() {};

  adjoint_source(
      type_real x, type_real y, type_real z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const std::string &station_name, const std::string &network_name)
      : vector_source(x, y, z, std::move(source_time_function)),
        station_name(station_name), network_name(network_name) {};

  adjoint_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : station_name(Node["station_name"].as<std::string>()),
        network_name(Node["network_name"].as<std::string>()),
        vector_source(Node, nsteps, dt) {};

  specfem::simulation::field_type get_wavefield_type() const override {
    return specfem::simulation::field_type::adjoint;
  }

  std::string source_name() const override { return "3-D adjoint source"; }

  /**
   * @brief Get the force vector
   *
   * Returns a unit force vector for adjoint source computations:
   *
   * \f[
   * \mathbf{f}_{adjoint} = \begin{cases}
   * [1.0] & \text{acoustic: unit pressure amplitude} \\
   * [1.0, 1.0, 1.0] & \text{elastic: unit forces in x,y,z directions}
   * \end{cases}
   * \f]
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Unit force vector with size depending on medium type
   */
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_force_vector() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

  std::string print_details() const override {
    std::ostringstream message;
    message << this->network_name << "." << this->station_name;
    return message.str();
  }

private:
  std::string station_name;
  std::string network_name;
};
} // namespace sources
} // namespace specfem
