#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source_time_functions.hpp"

#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Cosserat force source
 *
 * This class implements a Cosserat force source in 3D, which is used for
 * simulations in Cosserat elastic media. It combines both elastic and
 * rotational force components with separate scaling factors.
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
 * // Create a 3D Cosserat force source at (3.2, 4.8, 5.0)
 * auto cosserat_source =
 * specfem::sources::cosserat_force<specfem::element::dimension_tag::dim3>( 3.2,
 * // x-coordinate 4.8,  // y-coordinate 5.0,  // z-coordinate 0.5,  // fx -
 * elastic force in x direction 0.3,  // fy - elastic force in y direction 0.4,
 * // fz - elastic force in z direction 0.1,  // fc_x - rotational force in x
 * direction 0.2, // fc_y - rotational force in y direction 0.3,  // fc_z -
 * rotational force in z direction std::move(stf),
 * specfem::simulation::field_type::forward
 * );
 *
 * // Set the medium type (only works with Cosserat elastic media)
 * cosserat_source.set_medium_tag(specfem::element::medium_tag::elastic_spin);
 *
 * // Get the force vector (includes elastic and rotational components)
 * auto force_vector = cosserat_source.get_force_vector();
 * @endcode
 *
 */
template <>
class cosserat_force<specfem::element::dimension_tag::dim3>
    : public vector_source<specfem::element::dimension_tag::dim3> {

public:
  /**
   * @brief Default source constructor
   *
   */
  cosserat_force() {};
  /**
   * @brief Construct a new cosserat force object
   *
   * @param cosserat_source A YAML node defining cosserat force source
   * @param dt Time increment in the simulation. Used to calculate dominant
   * frequecy of Dirac source.
   */
  cosserat_force(YAML::Node &Node, const int nsteps, const type_real dt,
                 const specfem::simulation::field_type wavefield_type)
      : fx(Node["fx"].as<type_real>()), fy(Node["fy"].as<type_real>()),
        fz(Node["fz"].as<type_real>()), fc_x(Node["fc_x"].as<type_real>()),
        fc_y(Node["fc_y"].as<type_real>()), fc_z(Node["fc_z"].as<type_real>()),
        wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {
    // Store the parsed location as generic (coordinate-system) coordinates so
    // that source identity (operator==, which compares get_read_coordinates())
    // reflects the input coordinates. The base vector_source(Node, ...) ctor
    // only populates global_coordinates.
    this->set_read_coordinates(
        std::make_unique<specfem::coordinate_systems::cartesian_coordinates<
            specfem::element::dimension_tag::dim3>>(Node["x"].as<type_real>(),
                                                    Node["y"].as<type_real>(),
                                                    Node["z"].as<type_real>()));
  };

  type_real get_fx() const { return fx; }
  type_real get_fy() const { return fy; }
  type_real get_fz() const { return fz; }
  type_real get_fc_x() const { return fc_x; }
  type_real get_fc_y() const { return fc_y; }
  type_real get_fc_z() const { return fc_z; }
  /**
   * @brief Construct a new cosserat force object
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param fx Elastic force component in x-direction
   * @param fy Elastic force component in y-direction
   * @param fz Elastic force component in z-direction
   * @param fc_x Rotational force component in x-direction
   * @param fc_y Rotational force component in y-direction
   * @param fc_z Rotational force component in z-direction
   * @param source_time_function Pointer to source time function
   * @param wavefield_type Type of wavefield on which the source acts
   */
  cosserat_force(
      type_real x, type_real y, type_real z, type_real fx, type_real fy,
      type_real fz, type_real fc_x, type_real fc_y, type_real fc_z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : fx(fx), fy(fy), fz(fz), fc_x(fc_x), fc_y(fc_y), fc_z(fc_z),
        wavefield_type(wavefield_type),
        vector_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new cosserat force object from generic coordinates
   *
   * @param coordinates Generic coordinate object (resolved at assembly time)
   * @param fx Elastic force component in x-direction
   * @param fy Elastic force component in y-direction
   * @param fz Elastic force component in z-direction
   * @param fc_x Rotational force component in x-direction
   * @param fc_y Rotational force component in y-direction
   * @param fc_z Rotational force component in z-direction
   * @param source_time_function Pointer to source time function
   * @param wavefield_type Type of wavefield on which the source acts
   */
  cosserat_force(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real fx, type_real fy, type_real fz, type_real fc_x, type_real fc_y,
      type_real fc_z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : fx(fx), fy(fy), fz(fz), fc_x(fc_x), fc_y(fc_y), fc_z(fc_z),
        wavefield_type(wavefield_type),
        vector_source(std::move(coordinates), std::move(source_time_function)) {
        };

  std::string source_name() const override { return "3-D Cosserat force"; }

  /**
   * @brief User output
   *
   */
  std::string print_details() const override;

  specfem::simulation::field_type get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(
      const specfem::sources::source<specfem::element::dimension_tag::dim3>
          &other) const override;
  bool operator!=(
      const specfem::sources::source<specfem::element::dimension_tag::dim3>
          &other) const override;

  /**
   * @brief Get the force vector
   *
   * Returns the 3D Cosserat force vector combining elastic and rotational
   * components:
   *
   * \f[
   * \mathbf{f}_{Cosserat} = \begin{pmatrix}
   * f_x \\
   * f_y \\
   * f_z \\
   * f_{cx} \\
   * f_{cy} \\
   * f_{cz}
   * \end{pmatrix}
   * \f]
   *
   * This formulation is specific to Cosserat elastic media which include both
   * translational and rotational degrees of freedom.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Force vector with 6 components [fx, fy, fz, fc_x, fc_y, fc_z]
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
  static constexpr const char *name = "3-D Cosserat force";

private:
  type_real fx;   ///< Elastic force component in x-direction
  type_real fy;   ///< Elastic force component in y-direction
  type_real fz;   ///< Elastic force component in z-direction
  type_real fc_x; ///< Rotational force component in x-direction
  type_real fc_y; ///< Rotational force component in y-direction
  type_real fc_z; ///< Rotational force component in z-direction
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield on
                                                  ///< which the source
                                                  ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};

} // namespace sources
} // namespace specfem
