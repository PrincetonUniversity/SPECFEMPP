#pragma once

#include "fk_injection/config.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <optional>
#include <vector>

namespace fk_injection {

/**
 * @brief A single acoustic (fluid) layer in a 1-D velocity model.
 *
 * Depths are measured positive-downward from the top of the model.
 */
struct AcousticLayer {
  type_real density = 0;    ///< rho (kg/m^3)
  type_real p_velocity = 0; ///< vp (m/s)
  type_real thickness = 0;  ///< H (m)
};

/**
 * @brief A single elastic isotropic layer in a 1-D velocity model.
 *
 * The last elastic layer is the half-space; its @c thickness is ignored.
 */
struct ElasticIsotropicLayer {
  type_real density = 0;    ///< rho (kg/m^3)
  type_real p_velocity = 0; ///< vp (m/s)
  type_real s_velocity = 0; ///< vs (m/s)
  type_real thickness = 0;  ///< H (m); ignored for the half-space (last entry)
};

/**
 * @brief Result of @c LayeredModel::locate(): the resolved layer and geometry.
 *
 * All depth arithmetic is in metres, positive-downward.
 */
struct LayerLocation {
  specfem::plugin::Medium medium =
      specfem::plugin::Medium::elastic; ///< acoustic or elastic
  int layer = 0;     ///< 0-based index within the fluid block (acoustic) or
                     ///< elastic block (elastic)
  double height = 0; ///< height above the bottom of @c layer, m (>= 0; 0 at
                     ///< the layer's lower interface)
  bool in_halfspace = false; ///< true iff elastic and @c layer is the last
                             ///< elastic layer (the half-space)
  double depth_below_halfspace_top =
      0;                        ///< half-space top depth minus @c depth,
                                ///< m (<= 0 at/below the half-space top)
  bool at_free_surface = false; ///< acoustic layer 0 with depth within
                                ///< tolerance of 0
};

/**
 * @brief A 1-D horizontally-layered velocity model for FK synthesis.
 *
 * Stores a contiguous fluid (acoustic) block on top followed by a contiguous
 * elastic block below.  The fluid-above-solid invariant is structural: the
 * two blocks are held in separate vectors. The last entry of the elastic block
 * is the half-space.
 */
class LayeredModel {
public:
  /**
   * @brief Construct a layered model from typed fluid and elastic blocks.
   *
   * @param fluid_layers   Top-down contiguous acoustic block (may be empty).
   * @param elastic_layers Top-down elastic block below the fluid; the last
   *                       entry is the half-space (must be non-empty).
   * @throws std::runtime_error if @p elastic_layers is empty, if any density,
   *         p_velocity or elastic s_velocity is not positive, or if any layer
   *         above the half-space has a thickness that is not positive.
   */
  LayeredModel(std::vector<AcousticLayer> fluid_layers,
               std::vector<ElasticIsotropicLayer> elastic_layers);

  /**
   * @brief Return the fluid (acoustic) layer block.
   * @return const reference to the fluid layer vector
   */
  const std::vector<AcousticLayer> &fluid_layers() const;

  /**
   * @brief Return the elastic layer block.
   * @return const reference to the elastic layer vector
   */
  const std::vector<ElasticIsotropicLayer> &elastic_layers() const;

  /**
   * @brief Return the number of fluid (acoustic) layers.
   * @return fluid_layers_.size()
   */
  int number_of_fluid_layers() const;

  /**
   * @brief Return the number of elastic layers (including the half-space).
   * @return elastic_layers_.size()
   */
  int number_of_elastic_layers() const;

  /**
   * @brief Return whether the model contains at least one acoustic layer.
   * @return true if number_of_fluid_layers() > 0
   */
  bool has_fluid_layer() const;

  /**
   * @brief Locate a depth in the model and return full layer geometry.
   *
   * Points exactly on an interface are assigned to the shallower layer
   * (tie-goes-upper), ensuring continuity of the field state across interior
   * interfaces.
   *
   * @param depth      Depth in metres, positive-downward from the model top.
   * @param requested  Medium of the point; @c std::nullopt resolves it from
   *                   the layer table.
   * @param tolerance  Snap/match tolerance in metres (>= 0).
   * @return Populated @c LayerLocation.
   * @throws std::runtime_error if @p tolerance is negative or NaN, if @p depth
   *         is NaN or above the free surface, if @p requested is not acoustic
   *         or elastic, or if it is inconsistent with the model (e.g. acoustic
   *         requested on a model with no fluid layers).
   */
  LayerLocation locate(double depth,
                       std::optional<specfem::plugin::Medium> requested,
                       double tolerance) const;

  /**
   * @brief Return the total thickness of the fluid column in metres.
   * @return Sum of fluid-layer thicknesses; 0 if there are no fluid layers.
   */
  double fluid_thickness() const;

  /**
   * @brief Depth of the half-space top below the model top.
   * @return Total thickness of the fluid and finite elastic layers (m).
   */
  double halfspace_top_depth() const;

private:
  void validate() const;

  std::vector<AcousticLayer> fluid_layers_{}; ///< Fluid block, top to bottom
  std::vector<ElasticIsotropicLayer> elastic_layers_{}; ///< Elastic block, top
                                                        ///< to bottom
  std::vector<double> top_depths_{}; ///< Top depth of each layer; fluid layers
                                     ///< first, then elastic layers
};

} // namespace fk_injection
