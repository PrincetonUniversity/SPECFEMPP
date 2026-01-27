#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {
/**
 * @brief External source
 *
 * This class implements an external source in 2D, which is used for
 * coupling with external data or boundary conditions. External sources
 * provide a flexible interface for incorporating external forcing terms.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     25.0,  // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 2D external source at boundary location (0.0, 5.0)
 * auto ext_source = specfem::sources::external<specfem::dimension::type::dim2>(
 *     0.0,  // x-coordinate (at boundary)
 *     5.0,  // z-coordinate
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium type where the external source is located
 * ext_source.set_medium_tag(specfem::element::medium_tag::acoustic);
 *
 * // Get the force vector (unit vector components based on medium)
 * auto force_vector = ext_source.get_force_vector();
 * @endcode
 *
 */
template <>
class external<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

public:
  external() {};

  external(
      type_real x, type_real z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : vector_source(x, z, std::move(source_time_function)),
        wavefield_type(wavefield_type) {};

  external(YAML::Node &Node, const int nsteps, const type_real dt,
           const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {};

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  std::string print() const override;

  /**
   * @brief Get the force vector
   *
   * Returns a unit force vector for external boundary sources:
   *
   * \f[
   * \mathbf{f}_{external} = \begin{cases}
   * [1.0] & \text{acoustic: unit pressure amplitude} \\
   * [1.0] & \text{elastic SH: unit force out-of-plane} \\
   * [1.0, 1.0] & \text{elastic PSV: unit forces in x,z directions} \\
   * [1.0, 1.0, 1.0, 1.0] & \text{poroelastic: solid/fluid phases} \\
   * [1.0, 1.0] & \text{electromagnetic TE} \\
   * [1.0, 1.0, 1.0] & \text{elastic PSV-T: including rotation}
   * \end{cases}
   * \f]
   *
   * Where the unit components provide a normalized basis for external
   * coupling or boundary condition enforcement. The actual amplitudes
   * are scaled by the source time function during simulation.
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

public:
  static constexpr const char *name = "2-D external source";

private:
  specfem::wavefield::simulation_field wavefield_type;
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
