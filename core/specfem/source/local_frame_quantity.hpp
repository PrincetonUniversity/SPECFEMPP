#pragma once

#include <optional>
#include <stdexcept>
#include <variant>

namespace specfem {
namespace sources {

/**
 * @brief A source quantity read in a local frame and resolved to the global
 * Cartesian mesh frame at assembly time.
 *
 * Moment tensors and force vectors are supplied in a local geographic frame
 * (the spherical @f$ (r,\theta,\phi) @f$ basis for a moment tensor, the
 * East-North-Up basis for a force) whose orientation in the global Cartesian
 * mesh depends on the source location. This type mirrors the source coordinate
 * lifecycle — the as-read value is kept for provenance/output (like
 * `read_coordinates_`) while the resolved Cartesian value is filled once the
 * location fixes the local frame (like `global_coordinates`). A quantity read
 * directly in the Cartesian frame is resolved from construction.
 *
 * @tparam GeographicT The quantity in its as-read local (geographic) frame
 * @tparam CartesianT The quantity in the global Cartesian frame
 */
template <typename GeographicT, typename CartesianT> class LocalFrameQuantity {
public:
  LocalFrameQuantity() = default;

  /**
   * @brief Construct from a value already in the Cartesian frame (resolved).
   * @param cartesian The quantity in the global Cartesian frame
   */
  explicit LocalFrameQuantity(const CartesianT &cartesian)
      : input_(cartesian), cartesian_(cartesian) {}

  /**
   * @brief Construct from a value in the local geographic frame (unresolved).
   * @param geographic The quantity in its as-read local frame
   */
  explicit LocalFrameQuantity(const GeographicT &geographic)
      : input_(geographic), cartesian_(std::nullopt) {}

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
   * @brief The as-read value in its input frame (geographic or Cartesian).
   * @return Variant holding whichever frame the value was supplied in
   */
  const std::variant<GeographicT, CartesianT> &as_read() const {
    return input_;
  }

private:
  std::variant<GeographicT, CartesianT> input_; ///< As-read value (kept for
                                                ///< provenance)
  std::optional<CartesianT> cartesian_; ///< Resolved value; empty until set
};

} // namespace sources
} // namespace specfem
