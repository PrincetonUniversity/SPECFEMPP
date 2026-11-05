#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/layered_model.hpp"
#include "plane_wave_response.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <vector>

namespace fk_injection {

/** @brief A depth in a layer, which one series row serves. */
struct SeriesKey {
  int layer = 0;       ///< Index into @c LayeredModel::layers
  type_real depth = 0; ///< m, positive down from the model top
};

/**
 * @brief Depth-only series of the FK field, one row per (layer, depth).
 *
 * The field factorises as @f$u(x,y,z,t) = U(z, t - \tau(x,y))@f$, so a row
 * stores @f$U(z,\cdot)@f$ once for every point at its depth, as cubic-B-spline
 * coefficients laid out as @c impl::series_coefficient_index describes.
 *
 * Rows are mesh-exact: keys group only when their depths agree to float
 * round-off (see @c depth_tolerance), so a structured mesh's same-level points
 * share a row and distinct depth levels never merge. A sloped or curved
 * boundary gets mostly one row per point, which is still exact. A point on an
 * interface reads the row of the layer its material belongs to.
 */
class SeriesTable {
public:
  /**
   * @brief Construct from coefficients computed elsewhere.
   *
   * @param rows              Location of each row, sorted by layer and depth
   * @param coefficient_count Spline coefficients per series
   * @param coefficients      Row-major (row, series, coefficient)
   * @param tolerance         Depth slack of @c row, in m
   * @throws std::runtime_error if the coefficients don't match the counts, or
   *         the rows aren't sorted
   */
  SeriesTable(std::vector<LayerLocation> rows, int coefficient_count,
              std::vector<type_real> coefficients, double tolerance);

  /**
   * @brief Synthesize one row for each distinct (layer, depth) of @p keys.
   *
   * Each row is located at its shallowest key's depth, snapped to the layer's
   * interface within @p tolerance.
   *
   * @param response  Plane-wave response of the model
   * @param keys      Layers and depths the table must serve
   * @param tolerance Depth slack for grouping, snapping and @c row, in m
   * @return The table
   * @throws std::runtime_error if a key's depth lies outside its layer
   */
  static SeriesTable synthesize(const PlaneWaveResponse &response,
                                const std::vector<SeriesKey> &keys,
                                double tolerance);

  /**
   * @brief Float round-off of depths up to @p magnitude: 8 float ulps of it.
   *
   * Depths are narrowed to float, so depths the mesh placed at one level or
   * on one interface agree to this; no physical distance is involved.
   *
   * @param magnitude Largest coordinate magnitude in the mesh, m
   * @return Tolerance in m
   */
  static double depth_tolerance(double magnitude);

  int number_of_rows() const { return static_cast<int>(rows_.size()); }
  int coefficient_count() const { return coefficient_count_; }
  /// @brief Depth slack of @c row, in m.
  double tolerance() const { return tolerance_; }
  const LayerLocation &location(int row) const { return rows_[row]; }

  /**
   * @brief The row serving @p key.
   * @param key Layer and depth
   * @return Row index
   * @throws std::runtime_error if no row of that layer lies within
   *         @c tolerance above the depth
   */
  int row(const SeriesKey &key) const;

  /**
   * @brief Coefficients of one series of a row.
   * @param row    Row index
   * @param series Series slot, see @c impl::series_horizontal and others
   * @return Pointer to @c coefficient_count coefficients
   */
  const type_real *coefficients(int row, int series) const;

private:
  std::vector<LayerLocation> rows_{};
  std::vector<type_real> coefficients_{};
  int coefficient_count_ = 0;
  double tolerance_ = 0;
};

} // namespace fk_injection
