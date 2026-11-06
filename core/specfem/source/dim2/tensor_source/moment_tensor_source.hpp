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
 * @brief Moment-tensor source
 *
 * This class implements a moment tensor source in 2D, which represents
 * seismic sources like earthquakes through a symmetric stress tensor.
 * The moment tensor components Mxx, Mzz, and Mxz define the source mechanism.
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
 * // Create a 2D moment tensor source at (5.0, 8.0)
 * auto mt_source =
 * specfem::sources::moment_tensor<specfem::element::dimension_tag::dim2>( 5.0,
 * // x-coordinate 8.0,  // z-coordinate 1.0,  // Mxx - normal double couple in
 * x direction 2.0,  // Mzz - normal double couple in z direction 0.5,  // Mxz -
 * shear double couple in x-z plane std::move(stf),
 *     specfem::simulation::field_type::forward
 * );
 *
 * // Set the medium type (moment tensors work with elastic media)
 * mt_source.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Get the source tensor (2x2 symmetric matrix for 2D)
 * auto source_tensor = mt_source.get_source_tensor();
 * // source_tensor(0,0) = Mxx, source_tensor(0,1) = Mxz
 * // source_tensor(1,0) = Mxz, source_tensor(1,1) = Mzz
 * @endcode
 *
 */
template <>
class moment_tensor<specfem::element::dimension_tag::dim2>
    : public tensor_source<specfem::element::dimension_tag::dim2> {

public:
  /**
   * @brief Default source constructor
   *
   */
  moment_tensor() {};

  /**
   * @brief Get the Mxx component of the moment tensor
   *
   * @return type_real x-coordinate
   */
  type_real get_Mxx() const { return Mxx; }
  /**
   * @brief Get the Mxz component of the moment tensor
   *
   * @return type_real z-coordinate
   */
  type_real get_Mxz() const { return Mxz; }
  /**
   * @brief Get the Mzz component of the moment tensor
   *
   * @return type_real z-coordinate
   */
  type_real get_Mzz() const { return Mzz; }
  /**
   * @brief Get the Mzx component of the moment tensor
   *
   * For a symmetric (seismic) moment tensor this equals Mxz. An asymmetric
   * tensor (\f$ M_{xz} \neq M_{zx} \f$) drives the rotation field in 2D
   * Cosserat media via the body couple.
   *
   * @return type_real Mzx component
   */
  type_real get_Mzx() const { return Mzx; }

  /**
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
                const specfem::simulation::field_type wavefield_type)
      : Mxx(Node["Mxx"].as<type_real>()), Mzz(Node["Mzz"].as<type_real>()),
        Mxz(Node["Mxz"].as<type_real>()), Mzx([&Node]() -> type_real {
          // Optional: asymmetric tensors set Mzx; default to Mxz (symmetric).
          if (Node["Mzx"]) {
            return Node["Mzx"].as<type_real>();
          }
          return Node["Mxz"].as<type_real>();
        }()),
        wavefield_type(wavefield_type),
        tensor_source<specfem::element::dimension_tag::dim2>(Node, nsteps, dt) {
        };

  /**
   * @brief Costruct new moment tensor source using forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param Mxx Mxx component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   *
   * @note This overload builds a symmetric tensor (\f$ M_{zx} = M_{xz} \f$).
   */
  moment_tensor(
      type_real x, type_real z, const type_real Mxx, const type_real Mzz,
      const type_real Mxz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mxz), wavefield_type(wavefield_type),
        tensor_source<specfem::element::dimension_tag::dim2>(
            x, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new (possibly asymmetric) moment tensor source
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param Mxx Mxx component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param Mzx Mzx component of moment tensor (equals Mxz for a symmetric
   * tensor)
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  moment_tensor(
      type_real x, type_real z, const type_real Mxx, const type_real Mzz,
      const type_real Mxz, const type_real Mzx,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), Mzx(Mzx), wavefield_type(wavefield_type),
        tensor_source<specfem::element::dimension_tag::dim2>(
            x, z, std::move(source_time_function)) {};

  /**
   * @brief User output
   *
   */
  std::string source_name() const override { return "2-D moment tensor"; }
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
   * Returns the 2D seismic moment tensor for this source:
   *
   * \f[
   * \mathbf{M}_{2D} = \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * Where the components represent:
   * - \f$M_{xx}\f$: Normal stress component in x-direction
   * - \f$M_{zz}\f$: Normal stress component in z-direction
   * - \f$M_{xz}\f$: Shear stress component in x-z plane
   *
   * The tensor format depends on the medium type:
   *
   * **Elastic PSV** (2×2 matrix):
   * \f[
   * \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * **Elastic PSV-T (Cosserat)** (3×2 matrix). The rotation row of the tensor
   * is zero; the rotational forcing enters through the body couple instead
   * (see @ref get_body_couple_vector):
   * \f[
   * \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{zx} & M_{zz} \\
   * 0.0 & 0.0
   * \end{pmatrix}
   * \f]
   *
   * **Poroelastic** (4×2 matrix - duplicated for solid/fluid phases):
   * \f[
   * \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz} \\
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * **Electromagnetic TE** (2×2 matrix - same as elastic PSV):
   * \f[
   * \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Source tensor with dimensions [ncomponents][2] where each row contains
   * [Mxx, Mxz], [Mxz, Mzz] etc, depending on the medium type
   */
  Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_source_tensor() const override;

  /**
   * @brief Get the body-couple vector for the monopole source term
   *
   * The antisymmetric part of the moment tensor contracts (via the 2D
   * Levi-Civita symbol) to the scalar couple \f$ M_{xz} - M_{zx} \f$, which
   * drives the micro-rotation degree of freedom in Cosserat media. For
   * `elastic_psv_t` this returns the 3-component vector
   * \f$ [0, 0, M_{xz} - M_{zx}] \f$; a symmetric tensor therefore produces no
   * rotational coupling. For all other media (which have no rotational degree
   * of freedom) an empty view is returned.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Body-couple vector for `elastic_psv_t`, otherwise an empty view
   */
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_body_couple_vector() const override;

  /**
   * @brief Check if the moment tensor is asymmetric (i.e., Mxz != Mzx)
   *
   * @return true if the moment tensor is asymmetric, false otherwise
   */
  bool has_monopole_contribution() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  type_real Mxx; ///< Mxx for the source
  type_real Mxz; ///< Mxz for the source
  type_real Mzz; ///< Mzz for the source
  type_real Mzx; ///< Mzx for the source (defaults to Mxz: symmetric tensor)
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield on
                                                  ///< which the source
                                                  ///< acts

public:
protected:
};
} // namespace sources
} // namespace specfem
