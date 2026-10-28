#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/is_close.hpp"

#include <type_traits>

namespace specfem {
namespace sources {

/**
 * @brief Seismic moment tensor expressed in the axes of Frame.
 *
 * Full 3x3 storage (not Voigt-packed): a seismic tensor is symmetric, but a
 * Cosserat (`elastic_spin`) source may carry an antisymmetric part in the
 * Cartesian lower triangle that drives the micro-rotation field.
 *
 * Index meaning per frame:
 * - @ref specfem::coordinate_systems::CartesianFrame: axes (x, y, z), so
 *   `(0,0) = Mxx, (1,1) = Myy, (2,2) = Mzz, (0,1) = Mxy, (0,2) = Mxz,
 *   (1,2) = Myz` and the lower triangle holds `Myx, Mzx, Mzy`.
 * - @ref specfem::coordinate_systems::GeocentricFrame: axes
 *   @f$ (\theta, \phi, r) @f$, so `(0,0) = Mtt, (1,1) = Mpp, (2,2) = Mrr,
 *   (0,1) = Mtp, (0,2) = Mrt, (1,2) = Mrp` (the Harvard-CMT / CMTSOLUTION
 *   components reordered; see the CMTSOLUTION reader).
 *
 * @tparam Frame Coordinate frame of the components.
 */
template <typename Frame> class MomentTensor {
public:
  MomentTensor() = default;

  /**
   * @brief Construct a symmetric tensor from its six independent components.
   *
   * @param m00 Component (0,0)
   * @param m11 Component (1,1)
   * @param m22 Component (2,2)
   * @param m01 Component (0,1) = (1,0)
   * @param m02 Component (0,2) = (2,0)
   * @param m12 Component (1,2) = (2,1)
   */
  MomentTensor(type_real m00, type_real m11, type_real m22, type_real m01,
               type_real m02, type_real m12)
      : MomentTensor(m00, m11, m22, m01, m02, m12, m01, m02, m12) {}

  /**
   * @brief Construct a (possibly asymmetric) tensor from all nine components.
   *
   * @param m00 Component (0,0)
   * @param m11 Component (1,1)
   * @param m22 Component (2,2)
   * @param m01 Component (0,1)
   * @param m02 Component (0,2)
   * @param m12 Component (1,2)
   * @param m10 Component (1,0)
   * @param m20 Component (2,0)
   * @param m21 Component (2,1)
   */
  MomentTensor(type_real m00, type_real m11, type_real m22, type_real m01,
               type_real m02, type_real m12, type_real m10, type_real m20,
               type_real m21)
      : coefficients_{ { m00, m01, m02 },
                       { m10, m11, m12 },
                       { m20, m21, m22 } } {}

  /** @brief Component access with zero-based (row, column) indices. */
  type_real &operator()(int row, int column) {
    return coefficients_[row][column];
  }
  /** @brief Component access with zero-based (row, column) indices. */
  type_real operator()(int row, int column) const {
    return coefficients_[row][column];
  }

private:
  type_real coefficients_[3][3] = {};
};

/**
 * @brief Component-wise closeness of two moment tensors in the same frame.
 *
 * @tparam Frame Coordinate frame of both tensors.
 * @param a First tensor
 * @param b Second tensor
 * @return true if every component is close (see specfem::utilities::is_close)
 */
template <typename Frame>
bool is_close(const MomentTensor<Frame> &a, const MomentTensor<Frame> &b) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      if (!specfem::utilities::is_close(a(row, column), b(row, column))) {
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Rotate a moment tensor between distinct orthonormal bases.
 *
 * Applies the rank-2 rotation @f$ \mathbf{M}' = R\,\mathbf{M}\,R^{\mathsf{T}}
 * @f$ with @f$ R @f$ mapping source components to target components, e.g.
 * @ref specfem::coordinate_systems::rotation_matrix. The product is formed in
 * double precision and narrowed to `type_real` once. This reproduces the
 * closed-form CMTSOLUTION rotation in SPECFEM3D_GLOBE's `locate_sources.f90`.
 *
 * @tparam TargetFrame Coordinate frame of the result.
 * @tparam SourceFrame Coordinate frame of the input.
 * @param input Tensor in the source frame.
 * @param rotation Orthonormal matrix with target axes as rows and source axes
 * as columns.
 * @return The same tensor expressed in the target frame.
 */
template <typename TargetFrame, typename SourceFrame>
  requires(!std::is_same_v<TargetFrame, SourceFrame>)
MomentTensor<TargetFrame>
rotate(const MomentTensor<SourceFrame> &input,
       const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame,
                                                      double> &rotation) {
  // temporary = R * M
  double temporary[3][3] = {};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int inner = 0; inner < 3; ++inner) {
        temporary[row][column] +=
            rotation[row][inner] * static_cast<double>(input(inner, column));
      }
    }
  }
  // output = temporary * R^T
  MomentTensor<TargetFrame> output{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double sum = 0.0;
      for (int inner = 0; inner < 3; ++inner) {
        sum += temporary[row][inner] * rotation[column][inner];
      }
      output(row, column) = static_cast<type_real>(sum);
    }
  }
  return output;
}

} // namespace sources
} // namespace specfem
