#pragma once

#include "specfem/constants.hpp"

#include "specfem/enums.hpp"
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
 * @brief Spin-tensor source
 *
 * This class implements a spin tensor source in 2D: a moment tensor
 * \f$ M_c \f$ that drives the micro-rotation degree of freedom of Cosserat
 * media (`elastic_psv_t`) instead of the displacement ones. It flows through
 * the same gradient contraction as an ordinary moment tensor, but fills the
 * rotational row of the source tensor and leaves the displacement rows at
 * zero.
 *
 * In 2D the retained rotation is \f$ \omega_y \f$, so the spin tensor reduces
 * to its y-row restricted to the plane: the components \f$ M_{c,yx} \f$ and
 * \f$ M_{c,yz} \f$.
 *
 * Unlike an asymmetric moment tensor, a spin tensor never produces a
 * body-couple (monopole) contribution.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     8.0,   // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 2D spin tensor source at (5.0, 8.0)
 * auto spin_source =
 *     specfem::sources::spin_tensor<specfem::element::dimension_tag::dim2>(
 *         5.0,  // x-coordinate
 *         8.0,  // z-coordinate
 *         1.0,  // Mcyx component
 *         0.5,  // Mcyz component
 *         std::move(stf), specfem::simulation::field_type::forward);
 *
 * // Spin tensors act on Cosserat media only
 * spin_source.set_medium_tag(specfem::element::medium_tag::elastic_psv_t);
 *
 * // Get the source tensor (3x2 matrix; only the rotation row is non-zero)
 * auto source_tensor = spin_source.get_source_tensor();
 * // source_tensor(2,0) = Mcyx, source_tensor(2,1) = Mcyz
 * @endcode
 *
 */
template <>
class spin_tensor<specfem::element::dimension_tag::dim2>
    : public tensor_source<specfem::element::dimension_tag::dim2> {

public:
  /**
   * @brief Default source constructor
   *
   */
  spin_tensor() {};

  /**
   * @brief Get the Mcyx component of the spin tensor
   *
   * @return type_real Mcyx component
   */
  type_real get_Mcyx() const { return Mcyx; }
  /**
   * @brief Get the Mcyz component of the spin tensor
   *
   * @return type_real Mcyz component
   */
  type_real get_Mcyz() const { return Mcyz; }

  /**
   * @brief Construct a new spin tensor source object
   *
   * @param Node a spin_tensor data holder read from source file written in
   * .yml format
   * @param nsteps number of time steps
   * @param dt time step size
   * @param wavefield_type type of wavefield
   */
  spin_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
              const specfem::simulation::field_type wavefield_type)
      : Mcyx(Node["Mcyx"].as<type_real>()), Mcyz(Node["Mcyz"].as<type_real>()),
        wavefield_type(wavefield_type),
        tensor_source<specfem::element::dimension_tag::dim2>(Node, nsteps, dt) {
        };

  /**
   * @brief Construct new spin tensor source using forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param Mcyx Mcyx component of spin tensor
   * @param Mcyz Mcyz component of spin tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  spin_tensor(
      type_real x, type_real z, const type_real Mcyx, const type_real Mcyz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mcyx(Mcyx), Mcyz(Mcyz), wavefield_type(wavefield_type),
        tensor_source<specfem::element::dimension_tag::dim2>(
            x, z, std::move(source_time_function)) {};

  /**
   * @brief User output
   *
   */
  std::string source_name() const override { return "2-D spin tensor"; }
  std::string print_details() const override;

  specfem::simulation::field_type get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(
      const specfem::sources::source<specfem::element::dimension_tag::dim2>
          &other) const override;
  bool operator!=(
      const specfem::sources::source<specfem::element::dimension_tag::dim2>
          &other) const override;

  /**
   * @brief Get the source tensor
   *
   * Returns the 2D spin tensor for this source. Only Cosserat media
   * (`elastic_psv_t`, components \f$ [u_x, u_z, \omega_y] \f$) are supported;
   * the displacement rows are zero and the rotation row carries the spin
   * tensor:
   *
   * \f[
   * \begin{pmatrix}
   * 0 & 0 \\
   * 0 & 0 \\
   * M_{c,yx} & M_{c,yz}
   * \end{pmatrix}
   * \f]
   *
   * so the \f$ \omega_y \f$ forcing from the gradient contraction is
   * \f$ M_{c,yx} \, \partial_x L + M_{c,yz} \, \partial_z L \f$.
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Source tensor with dimensions [3][2] where only the rotation row
   * [Mcyx, Mcyz] is non-zero
   */
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_source_tensor() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  type_real Mcyx = 0.0; ///< Mcyx for the source
  type_real Mcyz = 0.0; ///< Mcyz for the source
  specfem::simulation::field_type wavefield_type =
      specfem::simulation::field_type::forward; ///< Type of wavefield on
                                                ///< which the source
                                                ///< acts
};
} // namespace sources
} // namespace specfem
