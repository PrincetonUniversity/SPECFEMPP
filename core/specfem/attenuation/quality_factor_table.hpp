#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>

namespace specfem {
namespace attenuation {

/**
 * @brief Quantized quality-factor table memoizing the attenuation fit.
 *
 * Q is quantized to buckets of width 1/specfem::constants::Q_RESOLUTION. The
 * first Q seen in a bucket represents it: later Q values in the same bucket
 * are replaced by it, and the (expensive) fit runs once per bucket rather
 * than once per GLL point. This bounds the number of fits by the number of
 * occupied buckets, independent of mesh size, which makes GLL-varying Q
 * models affordable. Ports SPECFEM3D_GLOBE's model_attenuation_storage, but
 * is independent of dimension and formulation: the fit is supplied by the
 * caller.
 *
 * The table is host-only setup state owned by its user; it is not a global.
 * Any path that fits Q uses it, including the property reader of a
 * Cartesian GLL model, so it is not globe-specific.
 *
 * @tparam FitResult Value returned by the fit for one Q.
 *
 * @code
 * specfem::attenuation::QualityFactorTable<Fit> table;
 * const auto fit = [&](type_real q) { return fit_quality_factor(q, ...); };
 * const type_real q = table.representative(Q);
 * const auto &result = table.fit(q, fit);
 * @endcode
 */
template <typename FitResult> class QualityFactorTable {
public:
  /**
   * @brief Create an empty table.
   * @param maximum When positive, Q values that are nonpositive, infinite or
   * above it are replaced by it before quantization, as in SPECFEM3D_GLOBE.
   * When nonpositive, Q is used as given.
   */
  explicit QualityFactorTable(const type_real maximum = 0)
      : maximum_(maximum) {}

  /**
   * @brief Apply the table's Q limit.
   * @param q Quality factor
   * @return q, or the maximum if one is set and q is outside (0, maximum]
   * @throws std::invalid_argument if q is NaN
   */
  type_real clip(const type_real q) const {
    if (std::isnan(q)) {
      throw std::invalid_argument("Quality factor must not be NaN");
    }
    if (maximum_ > 0 && (q <= 0 || q > maximum_)) {
      return maximum_;
    }
    return q;
  }

  /**
   * @brief Bucket index of a quality factor.
   * @param q Quality factor
   * @return \f$\max(1, \lfloor Q_\text{clip}\,\text{Q\_RESOLUTION}\rfloor)\f$
   */
  long long bucket(const type_real q) const {
    const auto key = static_cast<long long>(static_cast<double>(clip(q)) *
                                            specfem::constants::Q_RESOLUTION);
    return key < 1 ? 1 : key;
  }

  /**
   * @brief Return the representative Q of q's bucket, registering q if new.
   * @param q Quality factor
   * @return First clipped Q stored in q's bucket
   */
  type_real representative(const type_real q) {
    return table_.try_emplace(bucket(q), Entry{ clip(q), std::nullopt })
        .first->second.q;
  }

  /**
   * @brief Fit result for q's bucket, fitting its representative on a miss.
   * @tparam Fit Callable FitResult(type_real)
   * @param q Quality factor
   * @param fit_function Fit evaluated at the representative Q on a miss
   * @return Fit of the bucket's representative Q
   */
  template <typename Fit>
  const FitResult &fit(const type_real q, const Fit &fit_function) {
    auto &entry = table_.try_emplace(bucket(q), Entry{ clip(q), std::nullopt })
                      .first->second;
    if (!entry.fit) {
      entry.fit = fit_function(entry.q);
      ++fits_;
    }
    return *entry.fit;
  }

  /**
   * @brief Number of fits performed, i.e. table misses.
   * @return Fit count
   */
  int fits() const { return fits_; }

private:
  struct Entry {
    type_real q;                  ///< Representative Q of the bucket
    std::optional<FitResult> fit; ///< Fit of q, once requested
  };
  type_real maximum_ = 0;
  std::map<long long, Entry> table_;
  int fits_ = 0;
};

/**
 * @brief The bulk (Qkappa) and shear (Qmu) tables of one attenuation setup.
 *
 * Pass one instance to every container fitted in the same setup, so that
 * all of them agree on each bucket's representative Q and fit each bucket
 * once. Host-only: it holds a std::map, so keep it out of structs that are
 * copied into device kernels.
 *
 * @tparam FitResult Value returned by the fit for one Q.
 */
template <typename FitResult> struct QualityFactorTables {
  /**
   * @brief Create empty tables.
   * @param maximum Q limit applied by both tables (see QualityFactorTable)
   */
  explicit QualityFactorTables(const type_real maximum = 0)
      : bulk(maximum), shear(maximum) {}

  QualityFactorTable<FitResult> bulk;  ///< Table for Qkappa
  QualityFactorTable<FitResult> shear; ///< Table for Qmu

  /**
   * @brief Fits performed by both tables.
   * @return Sum of bulk and shear fit counts
   */
  int fits() const { return bulk.fits() + shear.fits(); }
};

} // namespace attenuation
} // namespace specfem
