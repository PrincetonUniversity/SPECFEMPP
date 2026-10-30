#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"

#include <optional>
#include <stdexcept>
#include <variant>

namespace specfem {
namespace sources {

/**
 * @brief A source quantity read in a local frame and resolved to the global
 * Cartesian mesh frame at assembly time.
 *
 * Moment tensors and force vectors may be supplied in the local geocentric
 * @f$ (\theta, \phi, r) @f$ frame (CMTSOLUTION, FORCESOLUTION), whose
 * orientation in the global Cartesian mesh depends on the source location.
 * This type mirrors the source coordinate lifecycle: the as-read value is kept
 * for provenance/output (like `read_coordinates_`) while the resolved
 * Cartesian value is filled once the location fixes the local frame (like
 * `global_coordinates`). A quantity read directly in the Cartesian frame is
 * resolved from construction.
 *
 * Only the Cartesian-frame instantiation of @p Quantity can be installed by
 * @ref resolve, so a value in the wrong frame is rejected at compile time.
 *
 * @tparam Quantity Frame-templated quantity, e.g. @ref MomentTensor or
 * @ref ForceVector
 */
template <template <typename> class Quantity> class LocalFrameQuantity {
public:
  /// The quantity in its as-read local (geocentric) frame
  using GeocentricT = Quantity<specfem::coordinate_systems::GeocentricFrame>;
  /// The quantity in the global Cartesian frame
  using CartesianT = Quantity<specfem::coordinate_systems::CartesianFrame>;

  LocalFrameQuantity() = default;

  /**
   * @brief Construct from a value already in the Cartesian frame (resolved).
   * @param cartesian The quantity in the global Cartesian frame
   */
  explicit LocalFrameQuantity(const CartesianT &cartesian)
      : input_(cartesian), cartesian_(cartesian) {}

  /**
   * @brief Construct from a value in the local geocentric frame (unresolved).
   * @param geocentric The quantity in its as-read local frame
   */
  explicit LocalFrameQuantity(const GeocentricT &geocentric)
      : input_(geocentric), cartesian_(std::nullopt) {}

  /**
   * @brief Whether the Cartesian value is available.
   * @return true once resolved
   */
  bool is_resolved() const { return cartesian_.has_value(); }

  /**
   * @brief Install the Cartesian value resolved from the local frame.
   *
   * The as-read value is left untouched, so provenance is preserved and a
   * repeated call is a safe overwrite rather than a re-rotation.
   *
   * @param cartesian The resolved Cartesian value
   */
  void resolve(const CartesianT &cartesian) { cartesian_ = cartesian; }

  /**
   * @brief The resolved Cartesian value.
   * @return The Cartesian value
   * @throws std::runtime_error if accessed before resolution
   */
  const CartesianT &cartesian() const {
    if (!cartesian_.has_value()) {
      throw std::runtime_error(
          "specfem::sources::LocalFrameQuantity: accessed before resolution; "
          "the source quantity must be rotated into the Cartesian mesh frame "
          "at assembly time first");
    }
    return *cartesian_;
  }

  /**
   * @brief The as-read value in its input frame (geocentric or Cartesian).
   * @return Variant holding whichever frame the value was supplied in
   */
  const std::variant<GeocentricT, CartesianT> &as_read() const {
    return input_;
  }

private:
  std::variant<GeocentricT, CartesianT> input_; ///< As-read value (kept for
                                                ///< provenance)
  std::optional<CartesianT> cartesian_; ///< Resolved value; empty until set
};

} // namespace sources
} // namespace specfem
