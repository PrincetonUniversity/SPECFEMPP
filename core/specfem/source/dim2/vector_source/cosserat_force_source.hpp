#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/macros.hpp"
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
 * @brief Cosserat force source
 *
 * This class implements a Cosserat force source in 2D, which is used for
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
 * // Create a 2D Cosserat force source at (3.2, 4.8)
 * auto cosserat_source =
 * specfem::sources::cosserat_force<specfem::dimension::type::dim2>( 3.2,   //
 * x-coordinate 4.8,   // z-coordinate 1.5,   // f - elastic force scaling
 * factor 0.8,   // fc - rotational force scaling factor 30.0,  // angle in
 * degrees std::move(stf), specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium type (only works with Cosserat elastic media)
 * cosserat_source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
 *
 * // Get the force vector (includes elastic and rotational components)
 * auto force_vector = cosserat_source.get_force_vector();
 * @endcode
 *
 */
template <>
class cosserat_force<specfem::dimension::type::dim2>
    : public vector_source<specfem::dimension::type::dim2> {

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
                 const specfem::wavefield::simulation_field wavefield_type)
      : angle([](YAML::Node &Node) -> type_real {
          if (Node["angle"]) {
            return Node["angle"].as<type_real>();
          } else {
            return 0.0;
          }
        }(Node)),
        f(Node["f"].as<type_real>()), fc(Node["fc"].as<type_real>()),
        wavefield_type(wavefield_type), vector_source(Node, nsteps, dt) {};

  type_real get_angle() const { return angle; }
  type_real get_f() const { return f; }
  type_real get_fc() const { return fc; }
  /**
   * @brief Construct a new cosserat force object
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param f Factor to scale the elastic force
   * @param fc Factor to scale the rotational force
   * @param angle Angle of the elastic force source
   * @param source_time_function Pointer to source time function
   * @param wavefield_type Type of wavefield on which the source acts
   */
  cosserat_force(
      type_real x, type_real z, type_real f, type_real fc, type_real angle,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : f(f), fc(fc), angle(angle), wavefield_type(wavefield_type),
        vector_source(x, z, std::move(source_time_function)) {};
  /**
   * @brief User output
   *
   */
  std::string print() const override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;
  bool operator!=(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;

  /**
   * @brief Get the force vector
   *
   * Returns the 2D Cosserat force vector combining elastic and rotational
   * components:
   *
   * \f[
   * \mathbf{f}_{Cosserat} = \begin{pmatrix}
   * f \sin(\theta) \\
   * -f \cos(\theta) \\
   * f_c
   * \end{pmatrix}
   * \f]
   *
   * Where:
   * - \f$f \sin(\theta)\f$: Elastic force component in x-direction
   * - \f$-f \cos(\theta)\f$: Elastic force component in z-direction
   * - \f$f_c\f$: Rotational (Cosserat) force component (couple stress)
   * - \f$\theta\f$ is the force angle
   *
   * This formulation is specific to Cosserat elastic media which include both
   * translational and rotational degrees of freedom.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector with 3 components [fx, fz, fc]
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
  static constexpr const char *name = "2-D Cosserat force";

private:
  type_real angle; ///< Angle of the elastic force source
  type_real f;     ///< Factor to scale the elastic force
  type_real fc;    ///< Factor to scale the rotational force
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
  const static std::vector<specfem::element::medium_tag> supported_media;
};

} // namespace sources
} // namespace specfem
