#pragma once

#include "specfem/constants.hpp"

#include "specfem/enums.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/source.hpp"
#include "specfem/source/local_frame_quantity.hpp"
#include "specfem/source/moment_tensor.hpp"
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
 * This class implements a moment tensor source in 3D, which represents
 * seismic sources like earthquakes through a 3x3 moment tensor. A seismic
 * tensor is symmetric and fully characterized by its six independent
 * components (Mxx, Myy, Mzz, Mxy, Mxz, Myz); the optional lower-triangle
 * components (Myx, Mzx, Mzy) describe an asymmetric tensor whose antisymmetric
 * part drives the micro-rotation field in Cosserat (`elastic_spin`) media.
 *
 * The tensor is held in a @ref specfem::sources::LocalFrameQuantity: it may be
 * supplied directly in the global Cartesian frame, or in the local geocentric
 * @f$ (\theta,\phi,r) @f$ frame (CMTSOLUTION), in which case it is kept
 * as-read and rotated into the Cartesian mesh frame at assembly time.
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
 * specfem::sources::moment_tensor<specfem::element::dimension_tag::dim3>( 10.0,
 * // x-coordinate 15.0,  // y-coordinate 20.0,  // z-coordinate 1.2,   // Mxx -
 * normal stress in x direction 0.8,   // Myy - normal stress in y direction
 *     1.5,   // Mzz - normal stress in z direction
 *     0.3,   // Mxy - shear stress component
 *     0.1,   // Mxz - shear stress component
 *     0.2,   // Myz - shear stress component
 *     std::move(stf),
 *     specfem::simulation::field_type::forward
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
class moment_tensor<specfem::element::dimension_tag::dim3>
    : public tensor_source<specfem::element::dimension_tag::dim3> {

  using MomentTensorQuantity =
      specfem::sources::LocalFrameQuantity<specfem::sources::MomentTensor>;
  using CartesianTensor = MomentTensorQuantity::CartesianT;
  using GeocentricTensor = MomentTensorQuantity::GeocentricT;

public:
  /**
   * @brief Default source constructor
   *
   */
  moment_tensor() {};

  /**
   * @brief Get the Mxx component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Mxx moment tensor component
   */
  type_real get_Mxx() const { return moment_.cartesian()(0, 0); }

  /**
   * @brief Get the Myy component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Myy moment tensor component
   */
  type_real get_Myy() const { return moment_.cartesian()(1, 1); }

  /**
   * @brief Get the Mzz component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Mzz moment tensor component
   */
  type_real get_Mzz() const { return moment_.cartesian()(2, 2); }

  /**
   * @brief Get the Mxy component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Mxy moment tensor component
   */
  type_real get_Mxy() const { return moment_.cartesian()(0, 1); }

  /**
   * @brief Get the Mxz component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Mxz moment tensor component
   */
  type_real get_Mxz() const { return moment_.cartesian()(0, 2); }

  /**
   * @brief Get the Myz component of the (resolved) Cartesian moment tensor
   *
   * @return type_real Myz moment tensor component
   */
  type_real get_Myz() const { return moment_.cartesian()(1, 2); }

  /**
   * @brief Install the Cartesian components resolved from the as-read frame.
   *
   * Marks the tensor resolved, after which @ref get_source_tensor returns the
   * Cartesian tensor. Called by the assembly-time resolution once the source
   * location fixes the local frame. The as-read tensor is kept.
   *
   * @param cartesian Moment tensor in the global Cartesian basis
   */
  void set_cartesian_components(const CartesianTensor &cartesian) {
    moment_.resolve(cartesian);
  }

  /**
   * @brief The moment tensor as a resolvable local-frame quantity.
   *
   * Exposes the as-read tensor and whether it has been resolved, so the
   * assembly stage can rotate an unresolved geographic tensor into the mesh
   * frame.
   *
   * @return Reference to the held local-frame quantity
   */
  const MomentTensorQuantity &moment() const { return moment_; }

  /**
   * @brief Get the Myx component of the (resolved) Cartesian moment tensor
   *
   * For a symmetric (seismic) moment tensor this equals Mxy. An asymmetric
   * tensor drives the micro-rotation field in 3D Cosserat media via the body
   * couple.
   *
   * @return type_real Myx moment tensor component
   */
  type_real get_Myx() const { return moment_.cartesian()(1, 0); }

  /**
   * @brief Get the Mzx component of the (resolved) Cartesian moment tensor
   *
   * For a symmetric (seismic) moment tensor this equals Mxz.
   *
   * @return type_real Mzx moment tensor component
   */
  type_real get_Mzx() const { return moment_.cartesian()(2, 0); }

  /**
   * @brief Get the Mzy component of the (resolved) Cartesian moment tensor
   *
   * For a symmetric (seismic) moment tensor this equals Myz.
   *
   * @return type_real Mzy moment tensor component
   */
  type_real get_Mzy() const { return moment_.cartesian()(2, 1); }

  /**
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
                const specfem::simulation::field_type wavefield_type)
      : moment_(read_cartesian_tensor(Node)), wavefield_type(wavefield_type),
        tensor_source(Node, nsteps, dt) {};

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
      const specfem::simulation::field_type wavefield_type)
      : moment_(CartesianTensor{ Mxx, Myy, Mzz, Mxy, Mxz, Myz }),
        wavefield_type(wavefield_type),
        tensor_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new (possibly asymmetric) moment tensor source
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param Mxx Mxx component of moment tensor
   * @param Myy Myy component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxy Mxy component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param Myz Myz component of moment tensor
   * @param Myx Myx component (equals Mxy for a symmetric tensor)
   * @param Mzx Mzx component (equals Mxz for a symmetric tensor)
   * @param Mzy Mzy component (equals Myz for a symmetric tensor)
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  moment_tensor(
      type_real x, type_real y, type_real z, type_real Mxx, type_real Myy,
      type_real Mzz, type_real Mxy, type_real Mxz, type_real Myz, type_real Myx,
      type_real Mzx, type_real Mzy,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : moment_(CartesianTensor{ Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy }),
        wavefield_type(wavefield_type),
        tensor_source(x, y, z, std::move(source_time_function)) {};

  /**
   * @brief Construct a new moment tensor source from generic coordinates
   *
   * @param coordinates Generic coordinate object
   * @param Mxx Mxx component of moment tensor
   * @param Myy Myy component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxy Mxy component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param Myz Myz component of moment tensor
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  moment_tensor(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real Mxx, type_real Myy, type_real Mzz, type_real Mxy, type_real Mxz,
      type_real Myz,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : moment_(CartesianTensor{ Mxx, Myy, Mzz, Mxy, Mxz, Myz }),
        wavefield_type(wavefield_type),
        tensor_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief Construct a new (possibly asymmetric) moment tensor source from
   * generic coordinates
   *
   * @param coordinates Generic coordinate object
   * @param Mxx Mxx component of moment tensor
   * @param Myy Myy component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxy Mxy component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param Myz Myz component of moment tensor
   * @param Myx Myx component (equals Mxy for a symmetric tensor)
   * @param Mzx Mzx component (equals Mxz for a symmetric tensor)
   * @param Mzy Mzy component (equals Myz for a symmetric tensor)
   * @param source_time_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  moment_tensor(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      type_real Mxx, type_real Myy, type_real Mzz, type_real Mxy, type_real Mxz,
      type_real Myz, type_real Myx, type_real Mzx, type_real Mzy,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : moment_(CartesianTensor{ Mxx, Myy, Mzz, Mxy, Mxz, Myz, Myx, Mzx, Mzy }),
        wavefield_type(wavefield_type),
        tensor_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief Construct a moment tensor source given in the local geocentric
   * basis.
   *
   * The tensor is stored unresolved: the local frame that rotates it into the
   * Cartesian mesh basis is only known once the source location is placed, so
   * @ref get_source_tensor throws until the assembly-time resolution calls
   * @ref set_cartesian_components. Used by the CMTSOLUTION reader for
   * (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp) moment tensors.
   *
   * @param coordinates Generic coordinate object
   * @param geocentric Moment tensor in the geocentric (theta, phi, r) basis
   * @param source_time_function Pointer to source time function
   * @param wavefield_type Type of wavefield
   */
  moment_tensor(
      std::unique_ptr<specfem::coordinate_systems::coordinates<
          specfem::element::dimension_tag::dim3>>
          coordinates,
      const GeocentricTensor &geocentric,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function,
      const specfem::simulation::field_type wavefield_type)
      : moment_(geocentric), wavefield_type(wavefield_type),
        tensor_source(std::move(coordinates), std::move(source_time_function)) {
        };

  /**
   * @brief User output
   *
   */
  std::string source_name() const override { return "3-D moment tensor"; }
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
   * @brief Get the body-couple vector for the monopole source term
   *
   * The antisymmetric part of the moment tensor contracts (via the 3D
   * Levi-Civita symbol) to the axial vector
   * \f$ [M_{yz} - M_{zy}, M_{zx} - M_{xz}, M_{xy} - M_{yx}] \f$, which drives
   * the micro-rotation degrees of freedom in Cosserat media. For `elastic_spin`
   * this returns the 6-component vector
   * \f$ [0, 0, 0, M_{yz} - M_{zy}, M_{zx} - M_{xz}, M_{xy} - M_{yx}] \f$; a
   * symmetric tensor therefore produces no rotational coupling. For all other
   * media (which have no rotational degree of freedom) an empty view is
   * returned.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
   * Body-couple vector for `elastic_spin`, otherwise an empty view
   */
  Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_body_couple_vector() const override;

  /**
   * @brief Whether this source contributes a monopole (body-couple) term
   *
   * @return true for `elastic_spin` media, false otherwise
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
  /**
   * @brief Read a Cartesian moment tensor from a YAML source node.
   *
   * The lower-triangle components (Myx, Mzx, Mzy) are optional; when absent
   * they default to their transposes, giving a symmetric tensor.
   *
   * @param Node YAML node holding the moment tensor components
   * @return The moment tensor in the global Cartesian basis
   */
  static CartesianTensor read_cartesian_tensor(const YAML::Node &Node);

  MomentTensorQuantity moment_; ///< As-read tensor + resolved Cartesian tensor
  specfem::simulation::field_type wavefield_type; ///< Type of wavefield on
                                                  ///< which the source
                                                  ///< acts

public:
protected:
};
} // namespace sources
} // namespace specfem
