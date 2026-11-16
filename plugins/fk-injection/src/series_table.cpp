#include "series_table.hpp"

#include "impl/depth_series.hpp"
#include "impl/series_layout.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fk_injection::series_table_impl {

/// Row order: elastic first, independent of the SDK's enum values, then depth.
std::pair<int, double> order(specfem::plugin::Medium medium, double depth);

} // namespace fk_injection::series_table_impl

std::pair<int, double>
fk_injection::series_table_impl::order(const specfem::plugin::Medium medium,
                                       const double depth) {
  return { medium == specfem::plugin::Medium::acoustic ? 1 : 0, depth };
}

fk_injection::SeriesTable::SeriesTable(
    std::vector<fk_injection::LayerLocation> rows, const int coefficient_count,
    std::vector<fk_injection::type_real> coefficients, const double tolerance)
    : rows_(std::move(rows)), coefficients_(std::move(coefficients)),
      coefficient_count_(coefficient_count), tolerance_(tolerance) {
  if (coefficient_count < 0) {
    std::ostringstream message;
    message << "SeriesTable: negative coefficient count " << coefficient_count;
    throw std::runtime_error(message.str());
  }
  const std::size_t expected_size = rows_.size() *
                                    fk_injection::impl::number_of_series *
                                    static_cast<std::size_t>(coefficient_count);
  if (coefficients_.size() != expected_size) {
    std::ostringstream message;
    message << "SeriesTable: " << coefficients_.size()
            << " coefficients, expected " << expected_size << " ("
            << rows_.size() << " rows, " << fk_injection::impl::number_of_series
            << " series, " << coefficient_count << " coefficients)";
    throw std::runtime_error(message.str());
  }
  const auto order = [](const fk_injection::LayerLocation &location) {
    return fk_injection::series_table_impl::order(location.medium,
                                                  location.depth);
  };
  for (std::size_t i = 1; i < rows_.size(); ++i) {
    if (order(rows_[i]) < order(rows_[i - 1])) {
      std::ostringstream message;
      message << "SeriesTable: row " << i
              << " is out of order; rows are sorted by medium, elastic first, "
                 "and depth";
      throw std::runtime_error(message.str());
    }
  }
}

fk_injection::SeriesTable fk_injection::SeriesTable::synthesize(
    const fk_injection::PlaneWaveResponse &response,
    const std::vector<fk_injection::SeriesKey> &keys) {
  const fk_injection::LayeredModel &model = response.model();
  const fk_injection::impl::FkSetup &setup = response.setup();
  const int number_of_keys = static_cast<int>(keys.size());

  // Float round-off at the deepest key: points the mesh placed at one depth
  // share a row, and no physical distance is involved.
  double max_depth = 0.0;
  for (const auto &key : keys)
    max_depth = std::max(max_depth, std::fabs(static_cast<double>(key.depth)));
  const float max_depth_float = static_cast<float>(max_depth);
  const double tolerance =
      8.0 * static_cast<double>(
                std::nextafterf(max_depth_float,
                                std::numeric_limits<float>::infinity()) -
                max_depth_float);

  // Rejects keys the model can't serve.
  for (const auto &key : keys)
    model.locate(static_cast<double>(key.depth), key.medium, tolerance);

  std::vector<int> order(number_of_keys);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return fk_injection::series_table_impl::order(
               keys[a].medium, static_cast<double>(keys[a].depth)) <
           fk_injection::series_table_impl::order(
               keys[b].medium, static_cast<double>(keys[b].depth));
  });

  // A row starts at the first key beyond its predecessor's anchor.
  std::vector<fk_injection::LayerLocation> rows;
  for (int k = 0; k < number_of_keys; ++k) {
    const fk_injection::SeriesKey &key = keys[order[k]];
    const double depth = static_cast<double>(key.depth);
    if (rows.empty() || key.medium != rows.back().medium ||
        depth - rows.back().depth > tolerance)
      rows.push_back(model.locate(depth, key.medium, tolerance));
  }

  const int fft_samples = setup.sizes.number_of_fft_samples;
  const int coefficient_count = setup.sizes.number_of_stored_frequencies;
  const int number_of_series = fk_injection::impl::number_of_series;
  std::vector<fk_injection::type_real> coefficients(
      rows.size() * number_of_series *
      static_cast<std::size_t>(coefficient_count));
  std::vector<std::complex<double>> spectrum(
      static_cast<std::size_t>(number_of_series) * fft_samples);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::fill(spectrum.begin(), spectrum.end(), std::complex<double>(0.0, 0.0));
    fk_injection::impl::depth_spectrum(rows[row], response.states(), model,
                                       setup, spectrum.data(), fft_samples);
    fk_injection::impl::to_time_series(
        spectrum.data(), setup,
        &coefficients[fk_injection::impl::series_coefficient_index(
            static_cast<int>(row), 0, 0, coefficient_count)]);
  }

  return fk_injection::SeriesTable(std::move(rows), coefficient_count,
                                   std::move(coefficients), tolerance);
}

int fk_injection::SeriesTable::row(const fk_injection::SeriesKey &key) const {
  const double depth = static_cast<double>(key.depth);
  // The last row at or above the depth, which is the only one that can hold it.
  const auto after = std::upper_bound(
      rows_.begin(), rows_.end(),
      fk_injection::series_table_impl::order(key.medium, depth),
      [](const std::pair<int, double> &wanted,
         const fk_injection::LayerLocation &location) {
        return wanted < fk_injection::series_table_impl::order(location.medium,
                                                               location.depth);
      });
  if (after == rows_.begin() || std::prev(after)->medium != key.medium ||
      depth - std::prev(after)->depth > tolerance_) {
    std::ostringstream message;
    message << "SeriesTable: no row serves depth " << depth << " in the "
            << (key.medium == specfem::plugin::Medium::acoustic ? "acoustic"
                                                                : "elastic")
            << " medium";
    throw std::runtime_error(message.str());
  }
  return static_cast<int>(std::prev(after) - rows_.begin());
}

const fk_injection::type_real *
fk_injection::SeriesTable::coefficients(const int row, const int series) const {
  return coefficients_.data() + fk_injection::impl::series_coefficient_index(
                                    row, series, 0, coefficient_count_);
}
