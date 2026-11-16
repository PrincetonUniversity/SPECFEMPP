#pragma once

#include "fk_injection/config.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <variant>
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
  int layer = 0;                        ///< Index into @c LayeredModel::layers
  double height = 0; ///< height above the bottom of @c layer, m (>= 0; 0 at
                     ///< the layer's lower interface)
  bool in_halfspace = false; ///< true iff @c layer is the half-space
  double depth_below_halfspace_top =
      0;                        ///< half-space top depth minus @c depth,
                                ///< m (<= 0 at/below the half-space top)
  bool at_free_surface = false; ///< acoustic, in layer 0, with depth within
                                ///< tolerance of 0
  double depth = 0;             ///< depth that was located, m
};

/// @brief One layer of a @c LayeredModel.
using Layer = std::variant<AcousticLayer, ElasticIsotropicLayer>;

/**
 * @brief A 1-D horizontally-layered velocity model for FK synthesis.
 *
 * Layers are ordered top-down; the last one is the elastic half-space. FK
 * supports a fluid block on top of a solid block only, which the constructor
 * checks.
 */
class LayeredModel {
public:
  /**
   * @brief Construct a layered model from its layers, top-down.
   *
   * @param layers Layers top-down; the last one is the half-space.
   * @throws std::runtime_error if the last layer isn't elastic, an acoustic
   *         layer lies below an elastic one, any density, p_velocity or
   *         elastic s_velocity is not positive, or any layer above the
   *         half-space has a thickness that is not positive.
   */
  explicit LayeredModel(std::vector<Layer> layers);

  /// @brief The layers, top-down.
  const std::vector<Layer> &layers() const { return layers_; }

  /// @brief Number of layers, including the half-space.
  int number_of_layers() const { return static_cast<int>(layers_.size()); }

  /// @brief The half-space, the last layer.
  const ElasticIsotropicLayer &halfspace() const {
    return std::get<ElasticIsotropicLayer>(layers_.back());
  }

  /**
   * @brief Locate a depth in the model and return full layer geometry.
   *
   * Points within @p tolerance of an interface between two layers of
   * @p medium are assigned to the shallower layer (tie-goes-upper); see
   * @c locate_in for the geometry.
   *
   * @param depth      Depth in metres, positive-downward from the model top.
   * @param medium     Medium of the point.
   * @param tolerance  Snap/match tolerance in metres (>= 0).
   * @return Populated @c LayerLocation.
   * @throws std::runtime_error if @p tolerance is negative or NaN, if @p depth
   *         is NaN or above the free surface, if @p medium is not acoustic
   *         or elastic, or if it is inconsistent with the model (e.g. acoustic
   *         requested on a model with no fluid layers).
   */
  LayerLocation locate(double depth, specfem::plugin::Medium medium,
                       double tolerance) const;

  /**
   * @brief Locate a depth within a given layer.
   *
   * A depth within @p tolerance of the layer's top or bottom snaps to that
   * interface. The height is 0 throughout the half-space.
   *
   * @param layer     Index into @c layers.
   * @param depth     Depth in metres, positive-downward from the model top.
   * @param tolerance Snap tolerance in metres (>= 0).
   * @return Populated @c LayerLocation.
   * @throws std::runtime_error if @p tolerance is negative or NaN, @p layer is
   *         out of range, or @p depth lies outside the layer by more than
   *         @p tolerance.
   */
  LayerLocation locate_in(int layer, double depth, double tolerance) const;

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
  std::vector<Layer> layers_{};      ///< Top to bottom
  int first_elastic_layer_ = 0;      ///< Number of fluid layers on top
  std::vector<double> top_depths_{}; ///< Top depth of each layer
};

} // namespace fk_injection
