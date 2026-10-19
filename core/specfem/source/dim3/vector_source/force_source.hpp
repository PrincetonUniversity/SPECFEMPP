#pragma once

#include "specfem/enums.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source/force_rotation.hpp"
#include "specfem/source/local_frame_quantity.hpp"
#include "specfem/source_time_functions.hpp"

#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Collocated force source
 *
 * This class implements a collocated force source in 3D that applies forces
 * in the x, y, and z directions at a specific location in the simulation
 * domain.
 *
 * The force is held in a @ref specfem::sources::LocalFrameQuantity: it may be
 * supplied directly in the global Cartesian frame, or in the local
 * East-North-Up (geographic) frame (FORCESOLUTION on a globe), in which case it
 * is kept as-read and rotated into the Cartesian mesh frame at assembly time.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     15.0,  // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 3D force source at (1.0, 2.0, 3.0) with force components
 * auto force_source =
 * specfem::sources::force<specfem::element::dimension_tag::dim3>( 1.0,  //
 * x-coordinate 2.0,  // y-coordinate 3.0,  // z-coordinate 0.7,  // fx - force
 * in x direction 0.0,  // fy - force in y direction 0.7,  // fz - force in z
 * direction std::move(stf), specfem::simulation::field_type::forward
 * );
 *
 * // Set the medium type
 * force_source.set_medium_tag(specfem::element::medium_tag::elastic);
 *
 * // Get the force vector
 * auto force_vector = force_source.get_force_vector();
 * @endcode
 *
 */
template <>
class force<specfem::element::dimension_tag::dim3>
    : public vector_source<specfem::element::dimension_tag::dim3> {

  using ForceQuantity =
      specfem::sources::LocalFrameQuantity<specfem::sources::GeographicForce,
                                           specfem::sources::CartesianForce>;

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
        const specfem::simulation::field_type wavefield_type)
      : force_(specfem::sources::CartesianForce{ Node["fx"].as<type_real>(),
                                                 Node["fy"].as<type_real>(),
                                                 Node["fz"].as<type_real>() }),
        wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {};

  /**
   * @brief Construct a new collocated force object
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(
      type_real x, type_real y, type_real z, type_real fx, type_real fy,
      type_real fz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : force_(specfem::sources::CartesianForce{ fx, fy, fz }),
        wavefield_type(wavefield_type),
        vector_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new collocated force object from generic coordinates
   *
   * @param coordinates Generic coordinate object
   * @param fx Force in x-direction
   * @param fy Force in y-direction
   * @param fz Force in z-direction
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real fx, type_real fy, type_real fz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : force_(specfem::sources::CartesianForce{ fx, fy, fz }),
        wavefield_type(wavefield_type),
        vector_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief Construct a collocated force given in the local East-North-Up frame.
   *
   * The force is stored unresolved: the local frame that rotates it into the
   * Cartesian mesh basis is only known once the source location is placed, so
   * @ref get_force_vector throws until the assembly-time resolution calls
   * @ref set_cartesian_components. Used by the FORCESOLUTION reader on a globe.
   *
   * @param coordinates Generic coordinate object
   * @param geographic Force in the local East-North-Up frame
   * @param source_time_function Pointer to source time function
   * @param wavefield_type Type of wavefield
   */
  force(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      const specfem::sources::GeographicForce &geographic,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : force_(geographic), wavefield_type(wavefield_type),
        vector_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief User output
   *
   */
  std::string source_name() const override { return "3-D force"; }
  std::string print_details() const override;

  specfem::simulation::field_type get_wavefield_type() const override {
    return wavefield_type;
  }

  /**
   * @brief Install the Cartesian components resolved from the as-read frame.
   *
   * Marks the force resolved, after which @ref get_force_vector returns the
   * Cartesian force. Called by the assembly-time resolution once the source
   * location fixes the local frame. The as-read force is kept.
   *
   * @param cartesian Force in the global Cartesian frame
   */
  void
  set_cartesian_components(const specfem::sources::CartesianForce &cartesian) {
    force_.resolve(cartesian);
  }

  /**
   * @brief The force as a resolvable local-frame quantity.
   *
   * Exposes the as-read force and whether it has been resolved, so the assembly
   * stage can rotate an unresolved geographic force into the mesh frame.
   *
   * @return Reference to the held local-frame quantity
   */
  const ForceQuantity &force_quantity() const { return force_; }

  /**
   * @brief Get the forcing function
   *
   */
  bool operator==(
      const specfem::sources::source<specfem::element::dimension_tag::dim3>
          &other) const override;
  bool operator!=(
      const specfem::sources::source<specfem::element::dimension_tag::dim3>
          &other) const override;

  /**
   * @brief Get the force vector
   *
   * Returns the (resolved) 3D Cartesian force vector for this source:
   *
   * \f[
   * \mathbf{f}_{3D} = \begin{cases}
   * [f_x] & \text{acoustic: pressure amplitude} \\
   * \begin{pmatrix} f_x \\ f_y \\ f_z \end{pmatrix} & \text{elastic: force
   * components}
   * \end{cases}
   * \f]
   *
   * Where:
   * - \f$f_x, f_y, f_z\f$ are the user-specified force components
   * - For acoustic media, only the \f$f_x\f$ component is used as pressure
   * source
   * - For elastic media, all three components define the force vector
   *
   * The force components are applied directly as body forces, representing
   * point sources with user-specified directional amplitudes.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Force vector with 3 components [fx, fy, fz]
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
private:
  ForceQuantity force_; ///< As-read force + resolved Cartesian force
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield on
                                                  ///< which the source
                                                  ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};
} // namespace sources
} // namespace specfem
