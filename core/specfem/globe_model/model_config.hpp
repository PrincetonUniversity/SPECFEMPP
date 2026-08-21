#pragma once

#include <stdexcept>
#include <string>

namespace specfem {
namespace globe_model {

/**
 * @brief The mesher's resolved model selection, read verbatim from the thin
 * database.
 *
 * Every field here is a decision the mesher made and wrote into the
 * MODEL_CONFIG block of `procNNNNNN_specfempp_database.bin` (see the record
 * layout at the top of `save_database_specfempp.F90`). None of them is a solver
 * constant, and none has a defensible default -- which is why the scalars below
 * are initialized to out-of-range values that @ref validate rejects rather than
 * to plausible ones. A silently defaulted `nex_xi` or attenuation band does not
 * fail; it produces different crustal smoothing and different \f$ \tau \f$ than
 * the mesher used.
 *
 * The model is identified by @ref model_name alone. The Fortran catalog
 * re-derives every model flag and every discontinuity radius from it, so the
 * database's `codes` and `flags` records are deliberately *not* represented
 * here: there is nowhere to put them, and `get_model_parameters()` would
 * overwrite them. They exist on disk as verification data only.
 *
 * This header is intentionally free of the Fortran ABI and of any link
 * dependency on `specfem::globe_model`, so the database reader can populate a
 * `ModelConfig` without pulling in the opt-in Fortran archive.
 */
struct ModelConfig {
  /** @brief Database record 6, `MODEL`. */
  std::string model_name;

  /** @brief Database record 2, `PLANET_TYPE`. 1 = IPLANET_EARTH. */
  int planet_type = 0;

  /// @name Database record 9: the mesh decomposition
  /// @{
  /// Not cosmetic -- `nex_xi` sets the element size the crustal models smooth
  /// over, so a wrong value yields plausible but wrong crust.
  int nchunks = 0;
  int nex_xi = 0;
  int nex_eta = 0;
  /// @}

  /// @name Database record 4: the physics flags
  /// @{
  /// All seven arrive in one record, so a partial read fails the Fortran
  /// record-length check rather than silently leaving these false.
  bool ellipticity = false;
  bool topography = false;
  bool oceans = false;
  bool attenuation = false;
  bool gravity = false;
  bool rotation = false;
  /// @}

  /// @name Database record 10: the attenuation period band, in seconds
  /// @{
  /// Consulted only when @ref attenuation is set. The mesher computes these in
  /// `rcp_set_compute_parameters`, which the evaluator deliberately does not
  /// call, so the database is the only source for them.
  double min_attenuation_period = 0.0;
  double max_attenuation_period = 0.0;
  /// @}

  /**
   * @brief Rejects a configuration that was never fully populated.
   *
   * Mirrors the Fortran's own argument checks so a missing field is reported
   * here, by name, instead of surfacing as a bare status code from
   * `globe_evaluator_init` -- or, worse, as a successful run against the wrong
   * model. The period band is only required when @ref attenuation is set,
   * matching `globe_evaluator_init`.
   *
   * @throws std::invalid_argument naming the offending field.
   */
  void validate() const {
    if (model_name.empty()) {
      throw std::invalid_argument("specfem::globe_model::ModelConfig: "
                                  "model_name is empty; it must come "
                                  "from the MODEL record of the mesh database");
    }
    require_positive(planet_type, "planet_type");
    require_positive(nchunks, "nchunks");
    require_positive(nex_xi, "nex_xi");
    require_positive(nex_eta, "nex_eta");

    if (!attenuation) {
      return;
    }

    if (min_attenuation_period <= 0.0 || max_attenuation_period <= 0.0 ||
        max_attenuation_period <= min_attenuation_period) {
      throw std::invalid_argument(
          "specfem::globe_model::ModelConfig: attenuation is enabled but the "
          "period band [" +
          std::to_string(min_attenuation_period) + ", " +
          std::to_string(max_attenuation_period) +
          "] is not a positive, increasing interval");
    }
  }

private:
  /** @brief Throws unless @p value was set to something usable. */
  static void require_positive(const int value, const std::string &field) {
    if (value <= 0) {
      throw std::invalid_argument(
          "specfem::globe_model::ModelConfig: " + field + " is " +
          std::to_string(value) +
          "; it must be read from the mesh database, which is the only place "
          "the mesher's value exists");
    }
  }
};

} // namespace globe_model
} // namespace specfem
