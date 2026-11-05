#include "field_sampler.hpp"

#include "fk_injection/utilities/cubic_bspline.hpp"
#include "impl/series_layout.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace fk_injection::field_sampler_impl {

/// Sets the traction factors of @p sample from the layer's material.
void set_lame_factors(const fk_injection::ElasticIsotropicLayer &layer,
                      fk_injection::PointSample &sample);

} // namespace fk_injection::field_sampler_impl

void fk_injection::field_sampler_impl::set_lame_factors(
    const fk_injection::ElasticIsotropicLayer &layer,
    fk_injection::PointSample &sample) {
  const double density = static_cast<double>(layer.density);
  const double p_velocity = static_cast<double>(layer.p_velocity);
  const double s_velocity = static_cast<double>(layer.s_velocity);
  const double xi = (s_velocity * s_velocity) / (p_velocity * p_velocity);
  const double mu = density * s_velocity * s_velocity;
  const double lambda = density * p_velocity * p_velocity - 2.0 * mu;
  sample.xi1 = static_cast<fk_injection::type_real>(1.0 - 2.0 * xi);
  sample.xim = static_cast<fk_injection::type_real>((1.0 - xi) * mu);
  sample.bulk =
      static_cast<fk_injection::type_real>(lambda / (2.0 * (lambda + mu)));
}

fk_injection::FieldSampler::FieldSampler(
    std::shared_ptr<const fk_injection::SeriesTable> series,
    std::vector<fk_injection::PointSample> points,
    const fk_injection::type_real cos_phi,
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
    const fk_injection::PointSample &point = points_[i];
    if (point.row < 0 || point.row >= series_->number_of_rows()) {
      std::ostringstream message;
      message << "FieldSampler: point " << i << " has row " << point.row
              << " outside [0, " << series_->number_of_rows() << ")";
      throw std::runtime_error(message.str());
    }
    if (point.medium != specfem::plugin::Medium::acoustic &&
        point.medium != specfem::plugin::Medium::elastic) {
      std::ostringstream message;
      message << "FieldSampler: point " << i
              << " has a medium other than acoustic or elastic";
      throw std::runtime_error(message.str());
    }
  }
}

fk_injection::FieldSampler::FieldSampler(
    const fk_injection::PlaneWaveResponse &response,
    std::shared_ptr<const fk_injection::SeriesTable> series,
    const std::vector<fk_injection::FieldPoint> &points)
    : FieldSampler(
          series, point_samples(response, *series, points),
          static_cast<fk_injection::type_real>(std::cos(response.setup().phi)),
          static_cast<fk_injection::type_real>(std::sin(response.setup().phi)),
          response.setup().sizes.resampling_rate) {}

std::vector<fk_injection::PointSample>
fk_injection::FieldSampler::point_samples(
    const fk_injection::PlaneWaveResponse &response,
    const fk_injection::SeriesTable &series,
    const std::vector<fk_injection::FieldPoint> &points) {
  std::vector<fk_injection::PointSample> samples(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const fk_injection::FieldPoint &point = points[i];
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    response.check_coverage(x, y);
    fk_injection::PointSample &sample = samples[i];
    sample.row = series.row({ point.layer, point.depth });
    sample.offset = response.abscissa_offset(x, y);
    const fk_injection::LayerLocation &location = series.location(sample.row);
    sample.medium = location.medium;
    if (const auto *solid = std::get_if<fk_injection::ElasticIsotropicLayer>(
            &response.model().layers()[location.layer]))
      fk_injection::field_sampler_impl::set_lame_factors(*solid, sample);
    sample.normal_x = point.normal_x;
    sample.normal_y = point.normal_y;
    sample.normal_z = point.normal_z;
  }
  return samples;
}

void fk_injection::FieldSampler::sample(
    const int first_step, const int number_of_steps,
    std::span<fk_injection::type_real> out) const {
  const int point_count = number_of_points();
  const std::size_t expected_size = static_cast<std::size_t>(point_count) *
                                    number_of_sample_components *
                                    static_cast<std::size_t>(number_of_steps);
  if (number_of_steps < 0 || out.size() != expected_size) {
    std::ostringstream message;
    message << "FieldSampler::sample: out has " << out.size() << " values for "
            << point_count << " points, " << number_of_sample_components
            << " components and " << number_of_steps << " steps";
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
      static_cast<std::size_t>(number_of_sample_components) * number_of_steps;

#ifdef FK_INJECTION_ENABLE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int ipt = 0; ipt < point_count; ++ipt) {
    for (int step = 0; step < number_of_steps; ++step) {
      const fk_injection::PointSample &point = points_[ipt];
      const bool is_acoustic =
          point.medium == specfem::plugin::Medium::acoustic;
      const double abscissa =
          static_cast<double>(first_step + step) / resampling + point.offset;
      fk_injection::type_real *point_out = output + ipt * stride + step;
      const auto evaluate = [&](const int slot) {
        return static_cast<double>(cubic::evaluate<fk_injection::type_real>(
            series.coefficients(point.row, slot), coefficient_count, abscissa));
      };

      const double horizontal = evaluate(fk_injection::impl::series_horizontal);
      const double vertical = evaluate(fk_injection::impl::series_vertical);
      point_out[(sample_kinematic_offset + 0) * number_of_steps] =
          static_cast<fk_injection::type_real>(cos_phi * horizontal);
      point_out[(sample_kinematic_offset + 1) * number_of_steps] =
          static_cast<fk_injection::type_real>(sin_phi * horizontal);
      point_out[(sample_kinematic_offset + 2) * number_of_steps] =
          static_cast<fk_injection::type_real>(vertical);

      double tx = 0.0;
      double ty = 0.0;
      double tz = 0.0;
      if (!is_acoustic) {
        const double sigma_zz = evaluate(fk_injection::impl::series_sigma_zz);
        const double sigma_rz = evaluate(fk_injection::impl::series_sigma_rz);
        const double dx = evaluate(fk_injection::impl::series_dx);
        const double xi1 = static_cast<double>(point.xi1);
        const double xim = static_cast<double>(point.xim);
        const double bulk = static_cast<double>(point.bulk);
        const double nx = static_cast<double>(point.normal_x);
        const double ny = static_cast<double>(point.normal_y);
        const double nz = static_cast<double>(point.normal_z);

        const double sigma_rr = xi1 * sigma_zz - 4.0 * xim * dx;
        const double sigma_tt = bulk * (sigma_rr + sigma_zz);
        const double txx =
            sigma_rr * cos_phi * cos_phi + sigma_tt * sin_phi * sin_phi;
        const double txy = cos_phi * sin_phi * (sigma_rr - sigma_tt);
        const double txz = sigma_rz * cos_phi;
        const double tyy =
            sigma_rr * sin_phi * sin_phi + sigma_tt * cos_phi * cos_phi;
        const double tyz = sigma_rz * sin_phi;
        const double tzz = sigma_zz;
        tx = txx * nx + txy * ny + txz * nz;
        ty = txy * nx + tyy * ny + tyz * nz;
        tz = txz * nx + tyz * ny + tzz * nz;
      }
      point_out[(sample_traction_offset + 0) * number_of_steps] =
          static_cast<fk_injection::type_real>(tx);
      point_out[(sample_traction_offset + 1) * number_of_steps] =
          static_cast<fk_injection::type_real>(ty);
      point_out[(sample_traction_offset + 2) * number_of_steps] =
          static_cast<fk_injection::type_real>(tz);

      const double potential_rate =
          is_acoustic ? evaluate(fk_injection::impl::series_potential_rate)
                      : 0.0;
      point_out[sample_potential_rate_offset * number_of_steps] =
          static_cast<fk_injection::type_real>(potential_rate);
    }
  }
}
