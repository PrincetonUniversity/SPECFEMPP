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
 * This class implements a moment tensor source in 3D, which represents
 * seismic sources like earthquakes through a symmetric 3x3 stress tensor.
 * The six independent components (Mxx, Myy, Mzz, Mxy, Mxz, Myz) fully
 * characterize the source mechanism.
 *
 * @par Usage Example
 * @code
 * // Create a Ricker wavelet source time function
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     12.0,  // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create a 3D moment tensor source at (10.0, 15.0, 20.0)
 * auto mt_source =
 * specfem::sources::moment_tensor<specfem::dimension::type::dim3>( 10.0,  //
 * x-coordinate 15.0,  // y-coordinate 20.0,  // z-coordinate 1.2,   // Mxx -
 * normal stress in x direction 0.8,   // Myy - normal stress in y direction
 *     1.5,   // Mzz - normal stress in z direction
 *     0.3,   // Mxy - shear stress component
 *     0.1,   // Mxz - shear stress component
 *     0.2,   // Myz - shear stress component
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium type (moment tensors work with elastic media)
 * mt_source.set_medium_tag(specfem::element::medium_tag::elastic);
 *
 * // Get the source tensor (3x3 symmetric matrix for 3D)
 * auto source_tensor = mt_source.get_source_tensor();
 * @endcode
 *
 */
template <>
class moment_tensor<specfem::dimension::type::dim3>
    : public tensor_source<specfem::dimension::type::dim3> {

public:
  /**
   * @brief Default source constructor
   *
   */
  moment_tensor() {};

  /**
   * @brief Get the Mxx component of the moment tensor
   *
   * @return type_real Mxx moment tensor component
   */
  type_real get_Mxx() const { return Mxx; }

  /**
   * @brief Get the Myy component of the moment tensor
   *
   * @return type_real Myy moment tensor component
   */
  type_real get_Myy() const { return Myy; }

  /**
   * @brief Get the Mzz component of the moment tensor
   *
   * @return type_real Mzz moment tensor component
   */
  type_real get_Mzz() const { return Mzz; }

  /**
   * @brief Get the Mxy component of the moment tensor
   *
   * @return type_real Mxy moment tensor component
   */
  type_real get_Mxy() const { return Mxy; }

  /**
   * @brief Get the Mxz component of the moment tensor
   *
   * @return type_real Mxz moment tensor component
   */
  type_real get_Mxz() const { return Mxz; }

  /**
   * @brief Get the Myz component of the moment tensor
   *
   * @return type_real Myz moment tensor component
   */
  type_real get_Myz() const { return Myz; }

  /**
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
                const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Node["Mxx"].as<type_real>()), Myy(Node["Myy"].as<type_real>()),
        Mzz(Node["Mzz"].as<type_real>()), Mxy(Node["Mxy"].as<type_real>()),
        Mxz(Node["Mxz"].as<type_real>()), Myz(Node["Myz"].as<type_real>()),
        wavefield_type(wavefield_type), tensor_source(Node, nsteps, dt) {};

  /**
   * @brief Costruct new moment tensor source using forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param Mxx Mxx component of moment tensor
   * @param Mzz Myy component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxy Mxy component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param Myz Myz component of moment tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   *
   */
  moment_tensor(
      type_real x, type_real y, type_real z, type_real Mxx, type_real Myy,
      type_real Mzz, type_real Mxy, type_real Mxz, type_real Myz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Mxx), Myy(Myy), Mzz(Mzz), Mxy(Mxy), Mxz(Mxz), Myz(Myz),
        wavefield_type(wavefield_type),
        tensor_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief User output
   *
   */
  std::string print() const override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(const specfem::sources::source<specfem::dimension::type::dim3>
                      &other) const override;
  bool operator!=(const specfem::sources::source<specfem::dimension::type::dim3>
                      &other) const override;

  /**
   * @brief Get the source tensor
   *
   * Returns the full 3D seismic moment tensor for this source:
   *
   * \f[
   * \mathbf{M}_{3D} = \begin{pmatrix}
   * M_{xx} & M_{xy} & M_{xz} \\
   * M_{xy} & M_{yy} & M_{yz} \\
   * M_{xz} & M_{yz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * Where the six independent components represent:
   * - \f$M_{xx}, M_{yy}, M_{zz}\f$: Normal stress components (diagonal)
   * - \f$M_{xy}, M_{xz}, M_{yz}\f$: Shear stress components (off-diagonal)
   *
   * The tensor format is a 3×3 symmetric matrix for elastic media, representing
   * the complete seismic moment tensor used in earthquake source modeling.
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Source tensor with dimensions [ncomponents][3] where each row contains
   * [Mxx, Mxy, Mxz], [Mxy, Myy, Myz], [Mxz, Myz, Mzz] etc, depending on the
   * medium type
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
  type_real Myy;                                       ///< Myy for the source
  type_real Mzz;                                       ///< Mzz for the source
  type_real Mxy;                                       ///< Mxy for the source
  type_real Mxz;                                       ///< Mxz for the source
  type_real Myz;                                       ///< Myz for the source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts

public:
  static constexpr const char *name = "3-D moment tensor";

protected:
};
} // namespace sources
} // namespace specfem
