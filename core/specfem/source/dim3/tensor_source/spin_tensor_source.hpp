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
 * This class implements a spin tensor source in 3D: a moment tensor
 * \f$ M_c \f$ that drives the micro-rotation degrees of freedom of Cosserat
 * media (`elastic_spin`, components \f$ [u_x, u_y, u_z, \phi_x, \phi_y,
 * \phi_z] \f$) instead of the displacement ones. It flows through the same
 * gradient contraction as an ordinary moment tensor, but fills the three
 * rotation rows of the source tensor and leaves the displacement rows at zero.
 *
 * The spin tensor may be asymmetric: the lower-triangle components
 * (`Mcyx`, `Mczx`, `Mczy`) default to their transposes when not specified.
 * Unlike an asymmetric moment tensor, a spin tensor never produces a
 * body-couple (monopole) contribution.
 *
 * @par Usage Example
 * @code
 * // Create a 3D spin tensor source at (10.0, 15.0, 20.0)
 * auto spin_source =
 *     specfem::sources::spin_tensor<specfem::element::dimension_tag::dim3>(
 *         10.0, 15.0, 20.0,
 *         1.2, 0.8, 1.5, 0.3, 0.1, 0.2, // Mcxx, Mcyy, Mczz, Mcxy, Mcxz, Mcyz
 *         std::move(stf), specfem::simulation::field_type::forward);
 *
 * // Spin tensors act on Cosserat media only
 * spin_source.set_medium_tag(specfem::element::medium_tag::elastic_spin);
 *
 * // Get the source tensor (6x3; only the rotation rows are non-zero)
 * auto source_tensor = spin_source.get_source_tensor();
 * @endcode
 *
 */
template <>
class spin_tensor<specfem::element::dimension_tag::dim3>
    : public tensor_source<specfem::element::dimension_tag::dim3> {

public:
  /**
   * @brief Default source constructor
   *
   */
  spin_tensor() {};

  /**
   * @brief Get the Mcxx component of the spin tensor
   * @return type_real Mcxx component
   */
  type_real get_Mcxx() const { return Mcxx; }
  /**
   * @brief Get the Mcyy component of the spin tensor
   * @return type_real Mcyy component
   */
  type_real get_Mcyy() const { return Mcyy; }
  /**
   * @brief Get the Mczz component of the spin tensor
   * @return type_real Mczz component
   */
  type_real get_Mczz() const { return Mczz; }
  /**
   * @brief Get the Mcxy component of the spin tensor
   * @return type_real Mcxy component
   */
  type_real get_Mcxy() const { return Mcxy; }
  /**
   * @brief Get the Mcxz component of the spin tensor
   * @return type_real Mcxz component
   */
  type_real get_Mcxz() const { return Mcxz; }
  /**
   * @brief Get the Mcyz component of the spin tensor
   * @return type_real Mcyz component
   */
  type_real get_Mcyz() const { return Mcyz; }
  /**
   * @brief Get the Mcyx component of the spin tensor (defaults to Mcxy)
   * @return type_real Mcyx component
   */
  type_real get_Mcyx() const { return Mcyx; }
  /**
   * @brief Get the Mczx component of the spin tensor (defaults to Mcxz)
   * @return type_real Mczx component
   */
  type_real get_Mczx() const { return Mczx; }
  /**
   * @brief Get the Mczy component of the spin tensor (defaults to Mcyz)
   * @return type_real Mczy component
   */
  type_real get_Mczy() const { return Mczy; }

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
      : Mcxx(Node["Mcxx"].as<type_real>()), Mcyy(Node["Mcyy"].as<type_real>()),
        Mczz(Node["Mczz"].as<type_real>()), Mcxy(Node["Mcxy"].as<type_real>()),
        Mcxz(Node["Mcxz"].as<type_real>()), Mcyz(Node["Mcyz"].as<type_real>()),
        // Optional: asymmetric tensors set the lower-triangle components;
        // default to their transpose (symmetric) when absent.
        Mcyx([&Node]() -> type_real {
          if (Node["Mcyx"]) {
            return Node["Mcyx"].as<type_real>();
          }
          return Node["Mcxy"].as<type_real>();
        }()),
        Mczx([&Node]() -> type_real {
          if (Node["Mczx"]) {
            return Node["Mczx"].as<type_real>();
          }
          return Node["Mcxz"].as<type_real>();
        }()),
        Mczy([&Node]() -> type_real {
          if (Node["Mczy"]) {
            return Node["Mczy"].as<type_real>();
          }
          return Node["Mcyz"].as<type_real>();
        }()),
        wavefield_type(wavefield_type), tensor_source(Node, nsteps, dt) {};

  /**
   * @brief Construct new (symmetric) spin tensor source using forcing function
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param Mcxx Mcxx component of spin tensor
   * @param Mcyy Mcyy component of spin tensor
   * @param Mczz Mczz component of spin tensor
   * @param Mcxy Mcxy component of spin tensor
   * @param Mcxz Mcxz component of spin tensor
   * @param Mcyz Mcyz component of spin tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   *
   * @note This overload builds a symmetric tensor (lower triangle equals the
   * upper triangle).
   */
  spin_tensor(
      type_real x, type_real y, type_real z, type_real Mcxx, type_real Mcyy,
      type_real Mczz, type_real Mcxy, type_real Mcxz, type_real Mcyz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz), Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz),
        Mcyx(Mcxy), Mczx(Mcxz), Mczy(Mcyz), wavefield_type(wavefield_type),
        tensor_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new (possibly asymmetric) spin tensor source
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param Mcxx Mcxx component of spin tensor
   * @param Mcyy Mcyy component of spin tensor
   * @param Mczz Mczz component of spin tensor
   * @param Mcxy Mcxy component of spin tensor
   * @param Mcxz Mcxz component of spin tensor
   * @param Mcyz Mcyz component of spin tensor
   * @param Mcyx Mcyx component (equals Mcxy for a symmetric tensor)
   * @param Mczx Mczx component (equals Mcxz for a symmetric tensor)
   * @param Mczy Mczy component (equals Mcyz for a symmetric tensor)
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  spin_tensor(
      type_real x, type_real y, type_real z, type_real Mcxx, type_real Mcyy,
      type_real Mczz, type_real Mcxy, type_real Mcxz, type_real Mcyz,
      type_real Mcyx, type_real Mczx, type_real Mczy,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz), Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz),
        Mcyx(Mcyx), Mczx(Mczx), Mczy(Mczy), wavefield_type(wavefield_type),
        tensor_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new (symmetric) spin tensor source from generic
   * coordinates
   *
   * @param coordinates Generic coordinate object
   * @param Mcxx Mcxx component of spin tensor
   * @param Mcyy Mcyy component of spin tensor
   * @param Mczz Mczz component of spin tensor
   * @param Mcxy Mcxy component of spin tensor
   * @param Mcxz Mcxz component of spin tensor
   * @param Mcyz Mcyz component of spin tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  spin_tensor(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real Mcxx, type_real Mcyy, type_real Mczz, type_real Mcxy,
      type_real Mcxz, type_real Mcyz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz), Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz),
        Mcyx(Mcxy), Mczx(Mcxz), Mczy(Mcyz), wavefield_type(wavefield_type),
        tensor_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief Construct a new (possibly asymmetric) spin tensor source from
   * generic coordinates
   *
   * @param coordinates Generic coordinate object
   * @param Mcxx Mcxx component of spin tensor
   * @param Mcyy Mcyy component of spin tensor
   * @param Mczz Mczz component of spin tensor
   * @param Mcxy Mcxy component of spin tensor
   * @param Mcxz Mcxz component of spin tensor
   * @param Mcyz Mcyz component of spin tensor
   * @param Mcyx Mcyx component (equals Mcxy for a symmetric tensor)
   * @param Mczx Mczx component (equals Mcxz for a symmetric tensor)
   * @param Mczy Mczy component (equals Mcyz for a symmetric tensor)
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  spin_tensor(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real Mcxx, type_real Mcyy, type_real Mczz, type_real Mcxy,
      type_real Mcxz, type_real Mcyz, type_real Mcyx, type_real Mczx,
      type_real Mczy,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mcxx(Mcxx), Mcyy(Mcyy), Mczz(Mczz), Mcxy(Mcxy), Mcxz(Mcxz), Mcyz(Mcyz),
        Mcyx(Mcyx), Mczx(Mczx), Mczy(Mczy), wavefield_type(wavefield_type),
        tensor_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief User output
   *
   */
  std::string source_name() const override { return "3-D spin tensor"; }
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
   * @brief Get the source tensor
   *
   * Returns the 3D spin tensor for this source. Only Cosserat media
   * (`elastic_spin`) are supported; the three displacement rows are zero and
   * the three rotation rows carry the spin tensor:
   *
   * \f[
   * \begin{pmatrix}
   * 0 & 0 & 0 \\
   * 0 & 0 & 0 \\
   * 0 & 0 & 0 \\
   * M_{c,xx} & M_{c,xy} & M_{c,xz} \\
   * M_{c,yx} & M_{c,yy} & M_{c,yz} \\
   * M_{c,zx} & M_{c,zy} & M_{c,zz}
   * \end{pmatrix}
   * \f]
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Source tensor with dimensions [6][3] where only the three rotation rows are
   * non-zero
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
  type_real Mcxx = 0.0; ///< Mcxx for the source
  type_real Mcyy = 0.0; ///< Mcyy for the source
  type_real Mczz = 0.0; ///< Mczz for the source
  type_real Mcxy = 0.0; ///< Mcxy for the source
  type_real Mcxz = 0.0; ///< Mcxz for the source
  type_real Mcyz = 0.0; ///< Mcyz for the source
  type_real Mcyx = 0.0; ///< Mcyx for the source (defaults to Mcxy)
  type_real Mczx = 0.0; ///< Mczx for the source (defaults to Mcxz)
  type_real Mczy = 0.0; ///< Mczy for the source (defaults to Mcyz)
  specfem::simulation::field_type wavefield_type =
      specfem::simulation::field_type::forward; ///< Type of wavefield on
                                                ///< which the source acts
};
} // namespace sources
} // namespace specfem
