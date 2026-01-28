#pragma once

#include "constants.hpp"
#include "enumerations/interface.hpp"
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
 * specfem::sources::moment_tensor<specfem::dimension::type::dim2>( 5.0,  //
 * x-coordinate 8.0,  // z-coordinate 1.0,  // Mxx - normal double couple in x
 * direction 2.0,  // Mzz - normal double couple in z direction 0.5,  // Mxz -
 * shear double couple in x-z plane std::move(stf),
 *     specfem::wavefield::simulation_field::forward
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
class moment_tensor<specfem::dimension::type::dim2>
    : public tensor_source<specfem::dimension::type::dim2> {

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
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
                const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Node["Mxx"].as<type_real>()), Mzz(Node["Mzz"].as<type_real>()),
        Mxz(Node["Mxz"].as<type_real>()), wavefield_type(wavefield_type),
        tensor_source<specfem::dimension::type::dim2>(Node, nsteps, dt) {};

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
   */
  moment_tensor(
      type_real x, type_real z, const type_real Mxx, const type_real Mzz,
      const type_real Mxz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), wavefield_type(wavefield_type),
        tensor_source<specfem::dimension::type::dim2>(
            x, z, std::move(source_time_function)) {};

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
   * **Elastic PSV-T (Cosserat)** (3×2 matrix):
   * \f[
   * \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz} \\
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
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  type_real Mxx;                                       ///< Mxx for the source
  type_real Mxz;                                       ///< Mxz for the source
  type_real Mzz;                                       ///< Mzz for the source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts

public:
  static constexpr const char *name = "2-D moment tensor";

protected:
};
} // namespace sources
} // namespace specfem
