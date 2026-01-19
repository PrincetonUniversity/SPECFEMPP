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
 * This class implements a collocated force source in 3D that applies forces
 * in the x, y, and z directions at a specific location in the simulation
 * domain.
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
 * auto force_source = specfem::sources::force<specfem::dimension::type::dim3>(
 *     1.0,  // x-coordinate
 *     2.0,  // y-coordinate
 *     3.0,  // z-coordinate
 *     0.7,  // fx - force in x direction
 *     0.0,  // fy - force in y direction
 *     0.7,  // fz - force in z direction
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
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
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  force(
      type_real x, type_real y, type_real z, type_real fx, type_real fy,
      type_real fz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : wavefield_type(wavefield_type),
        vector_source(x, y, z, std::move(source_time_function)), fx(fx), fy(fy),
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
   * Returns the 3D force vector for this source:
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
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector with 3 components [fx, fy, fz]
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
