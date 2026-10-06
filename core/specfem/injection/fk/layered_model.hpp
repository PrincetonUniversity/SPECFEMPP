#pragma once

#include "specfem/setup.hpp"
#include <stdexcept>
#include <vector>

namespace specfem {
namespace injection {
namespace fk {

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
 * The last elastic layer is the half-space; its @c thickness is ignored
 * by downstream propagators but is still accumulated in the top-depth
 * computation for simplicity.
 */
struct ElasticIsotropicLayer {
  type_real density = 0;    ///< rho (kg/m^3)
  type_real p_velocity = 0; ///< vp (m/s)
  type_real s_velocity = 0; ///< vs (m/s)
  type_real thickness = 0;  ///< H (m); ignored for the half-space (last entry)
};

/**
 * @brief A 1-D horizontally-layered velocity model for FK synthesis.
 *
 * Stores a contiguous fluid (acoustic) block on top followed by a contiguous
 * elastic block below.  The fluid-above-solid invariant is structural: the
 * two blocks are held in separate vectors and validated at construction time.
 * The last entry of the elastic block is the half-space.
 *
 * On construction the object computes cumulative top depths in global
 * top-to-bottom order (all fluid layers, then all elastic layers) and then
 * calls @c validate().
 */
class LayeredModel {
public:
  /** @brief Default-construct an empty model. */
  LayeredModel() = default;

  /**
   * @brief Construct a layered model from typed fluid and elastic blocks.
   *
   * Computes cumulative top depths in global order and calls @c validate().
   *
   * @param fluid_layers   Top-down contiguous acoustic block (may be empty).
   * @param elastic_layers Top-down elastic block below the fluid; the last
   *                       entry is the half-space (must be non-empty).
   * @throws std::runtime_error if validation fails.
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
   * @brief Return the total number of layers (fluid + elastic).
   * @return number_of_fluid_layers() + number_of_elastic_layers()
   */
  int total_number_of_layers() const;

  /**
   * @brief Return whether the model contains at least one acoustic layer.
   * @return true if number_of_fluid_layers() > 0
   */
  bool has_fluid_layer() const;

  /**
   * @brief Compute the shear modulus of an elastic layer.
   *
   * \f$ \mu = \rho \, v_s^2 \f$
   *
   * @param elastic_index 0-based index into the elastic layer block
   * @return shear modulus in Pa
   */
  type_real shear_modulus(int elastic_index) const;

  /**
   * @brief Return the precomputed top depth of each layer in global order.
   *
   * Global order is all fluid layers first, then all elastic layers.
   * @c layer_top_depths()[0] == 0; subsequent entries accumulate thicknesses.
   * Length equals @c total_number_of_layers().
   *
   * @return const reference to the top-depth vector
   */
  const std::vector<type_real> &layer_top_depths() const;

  /**
   * @brief Validate the model.
   *
   * Throws @c std::runtime_error if:
   *   - @c elastic_layers_ is empty (no half-space),
   *   - any layer has density <= 0 or p_velocity <= 0,
   *   - any elastic layer has s_velocity <= 0.
   *
   * @throws std::runtime_error on any validation failure
   */
  void validate() const;

private:
  std::vector<AcousticLayer> fluid_layers_{}; ///< Fluid block, top to bottom
  std::vector<ElasticIsotropicLayer> elastic_layers_{}; ///< Elastic block, top
                                                        ///< to bottom
  std::vector<type_real> top_depths_{}; ///< Cumulative top depth, global order
};

} // namespace fk
} // namespace injection
} // namespace specfem
