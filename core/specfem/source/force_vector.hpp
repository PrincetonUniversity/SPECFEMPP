#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/is_close.hpp"

#include <type_traits>

namespace specfem {
namespace sources {

/**
 * @brief Collocated force vector expressed in the axes of Frame.
 *
 * Component order follows the frame:
 * @ref specfem::coordinate_systems::CartesianFrame gives (fx, fy, fz);
 * @ref specfem::coordinate_systems::GeocentricFrame gives
 * @f$ (f_\theta, f_\phi, f_r) @f$. A FORCESOLUTION East/North/Up direction is
 * reordered into the geocentric frame by
 * @ref specfem::coordinate_systems::to_geocentric_components.
 *
 * @tparam Frame Coordinate frame of the components.
 */
template <typename Frame> class ForceVector {
public:
  ForceVector() = default;

  /**
   * @brief Construct from three components in frame order.
   *
   * @param c0 Component 0
   * @param c1 Component 1
   * @param c2 Component 2
   */
  ForceVector(type_real c0, type_real c1, type_real c2)
      : components_{ c0, c1, c2 } {}

  /** @brief Component access with a zero-based index. */
  type_real &operator[](int index) { return components_[index]; }
  /** @brief Component access with a zero-based index. */
  type_real operator[](int index) const { return components_[index]; }

private:
  type_real components_[3] = {};
};

/**
 * @brief Component-wise closeness of two force vectors in the same frame.
 *
 * @tparam Frame Coordinate frame of both vectors.
 * @param a First vector
 * @param b Second vector
 * @return true if every component is close (see specfem::utilities::is_close)
 */
template <typename Frame>
bool is_close(const ForceVector<Frame> &a, const ForceVector<Frame> &b) {
  for (int index = 0; index < 3; ++index) {
    if (!specfem::utilities::is_close(a[index], b[index])) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Rotate a force vector between distinct orthonormal bases.
 *
 * Applies the rank-1 rotation @f$ \mathbf{f}' = R\,\mathbf{f} @f$ with
 * @f$ R @f$ mapping source components to target components, e.g.
 * @ref specfem::coordinate_systems::rotation_matrix. The product is formed in
 * double precision and narrowed to `type_real` once.
 *
 * @tparam TargetFrame Coordinate frame of the result.
 * @tparam SourceFrame Coordinate frame of the input.
 * @param input Vector in the source frame.
 * @param rotation Orthonormal matrix with target axes as rows and source axes
 * as columns.
 * @return The same vector expressed in the target frame.
 */
template <typename TargetFrame, typename SourceFrame>
  requires(!std::is_same_v<TargetFrame, SourceFrame>)
ForceVector<TargetFrame>
rotate(const ForceVector<SourceFrame> &input,
       const specfem::coordinate_systems::BasisMatrix<TargetFrame, SourceFrame,
                                                      double> &rotation) {
  ForceVector<TargetFrame> output{};
  for (int row = 0; row < 3; ++row) {
    double sum = 0.0;
    for (int inner = 0; inner < 3; ++inner) {
      sum += rotation[row][inner] * static_cast<double>(input[inner]);
    }
    output[row] = static_cast<type_real>(sum);
  }
  return output;
}

} // namespace sources
} // namespace specfem
