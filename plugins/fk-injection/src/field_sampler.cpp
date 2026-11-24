#include "field_sampler.hpp"

#include "fk_injection/utilities/cubic_bspline.hpp"
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

template <typename MediumType>
fk_injection::FieldSampler<MediumType>::FieldSampler(
    std::shared_ptr<const fk_injection::SeriesTable> series,
    std::vector<Point> points, const fk_injection::type_real cos_phi,
    const fk_injection::type_real sin_phi, const int resampling_rate)
    : series_(std::move(series)), points_(std::move(points)), cos_phi_(cos_phi),
      sin_phi_(sin_phi), resampling_rate_(resampling_rate) {
  if (!series_)
    throw std::runtime_error("FieldSampler: no series table");
  if (resampling_rate < 1) {
    std::ostringstream message;
    message << "FieldSampler: resampling_rate must be at least 1 ("
            << resampling_rate << ")";
    throw std::runtime_error(message.str());
  }
  for (int i = 0; i < number_of_points(); ++i) {
    const int row = points_[i].row;
    if (row < 0 || row >= series_->number_of_rows()) {
      std::ostringstream message;
      message << "FieldSampler: point " << i << " has row " << row
              << " outside [0, " << series_->number_of_rows() << ")";
      throw std::runtime_error(message.str());
    }
    if (series_->location(row).medium != MediumType::medium) {
      std::ostringstream message;
      message << "FieldSampler: point " << i
              << " reads a row of another medium";
      throw std::runtime_error(message.str());
    }
  }
}

template <typename MediumType>
fk_injection::FieldSampler<MediumType>::FieldSampler(
    const fk_injection::PlaneWaveResponse &response,
    std::shared_ptr<const fk_injection::SeriesTable> series,
    const std::vector<fk_injection::FieldPoint> &points)
    : FieldSampler(
          series, point_samples(response, *series, points),
          static_cast<fk_injection::type_real>(std::cos(response.setup().phi)),
          static_cast<fk_injection::type_real>(std::sin(response.setup().phi)),
          response.setup().sizes.resampling_rate) {}

template <typename MediumType>
std::vector<typename MediumType::Point>
fk_injection::FieldSampler<MediumType>::point_samples(
    const fk_injection::PlaneWaveResponse &response,
    const fk_injection::SeriesTable &series,
    const std::vector<fk_injection::FieldPoint> &points) {
  std::vector<Point> samples;
  samples.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const fk_injection::FieldPoint &point = points[i];
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    response.check_coverage(x, y);
    const int row = series.row({ point.layer, point.depth });
    const fk_injection::LayerLocation &location = series.location(row);
    // A row of another medium is left for the constructor to reject.
    const auto *layer = std::get_if<typename MediumType::ModelLayer>(
        &response.model().layers()[location.layer]);
    Point &sample = samples.emplace_back(
        layer ? MediumType::point(*layer, point.normal_x, point.normal_y,
                                  point.normal_z)
              : Point{});
    sample.row = row;
    sample.offset = response.abscissa_offset(x, y);
  }
  return samples;
}

template <typename MediumType>
void fk_injection::FieldSampler<MediumType>::sample(
    const specfem::plugin::Quantity quantity, const int first_step,
    const int number_of_steps, std::span<fk_injection::type_real> out) const {
  if (!fk_injection::serves<MediumType>(quantity))
    throw std::runtime_error(
        "FieldSampler::sample: quantity not served by this medium");
  const int point_count = number_of_points();
  const int components = fk_injection::number_of_components(quantity);
  const std::size_t expected_size = static_cast<std::size_t>(point_count) *
                                    components *
                                    static_cast<std::size_t>(number_of_steps);
  if (number_of_steps < 0 || out.size() != expected_size) {
    std::ostringstream message;
    message << "FieldSampler::sample: out has " << out.size() << " values for "
            << point_count << " points, " << components << " components and "
            << number_of_steps << " steps";
    throw std::runtime_error(message.str());
  }
  if (point_count == 0)
    return;

  namespace cubic = fk_injection::cubic_bspline;

  const fk_injection::SeriesTable &series = *series_;
  fk_injection::type_real *output = out.data();
  const int coefficient_count = series.coefficient_count();
  const double resampling = static_cast<double>(resampling_rate_);
  const double cos_phi = static_cast<double>(cos_phi_);
  const double sin_phi = static_cast<double>(sin_phi_);
  const std::size_t stride =
      static_cast<std::size_t>(components) * number_of_steps;

#ifdef FK_INJECTION_ENABLE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int ipt = 0; ipt < point_count; ++ipt) {
    for (int step = 0; step < number_of_steps; ++step) {
      const Point &point = points_[ipt];
      const double abscissa =
          static_cast<double>(first_step + step) / resampling + point.offset;
      const auto evaluate = [&](const int series_index) {
        return static_cast<double>(cubic::evaluate<fk_injection::type_real>(
            series.coefficients(point.row, series_index), coefficient_count,
            abscissa));
      };
      const auto values =
          MediumType::project(quantity, point, cos_phi, sin_phi, evaluate);
      fk_injection::type_real *point_out = output + ipt * stride + step;
      for (int component = 0; component < components; ++component)
        point_out[component * number_of_steps] =
            static_cast<fk_injection::type_real>(values[component]);
    }
  }
}

template class fk_injection::FieldSampler<fk_injection::ElasticMedium>;
template class fk_injection::FieldSampler<fk_injection::AcousticMedium>;
