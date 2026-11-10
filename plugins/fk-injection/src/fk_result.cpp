#include "fk_injection/fk_result.hpp"

#include "fk_injection/utilities/cubic_bspline.hpp"
#include "impl/series_layout.hpp" // series slot indices
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

fk_injection::FkResult::FkResult(std::vector<type_real> series_coefficients,
                                 int number_of_groups, int coefficient_count,
                                 std::vector<PointSample> points,
                                 type_real cos_phi, type_real sin_phi,
                                 int resampling_rate)
    : series_coefficients_(std::move(series_coefficients)),
      points_(std::move(points)), cos_phi_(cos_phi), sin_phi_(sin_phi),
      resampling_rate_(resampling_rate), number_of_groups_(number_of_groups),
      coefficient_count_(coefficient_count) {
  if (number_of_groups < 0 || coefficient_count < 0) {
    std::ostringstream msg;
    msg << "FkResult: negative shape (" << number_of_groups << " groups, "
        << coefficient_count << " coefficients).";
    throw std::runtime_error(msg.str());
  }
  const std::size_t expected_size = static_cast<std::size_t>(number_of_groups) *
                                    fk_injection::impl::number_of_series *
                                    coefficient_count;
  if (series_coefficients_.size() != expected_size) {
    std::ostringstream msg;
    msg << "FkResult: series_coefficients has " << series_coefficients_.size()
        << " values, expected " << expected_size << " (" << number_of_groups
        << " groups, " << fk_injection::impl::number_of_series << " series, "
        << coefficient_count << " coefficients).";
    throw std::runtime_error(msg.str());
  }
  if (resampling_rate < 1) {
    std::ostringstream msg;
    msg << "FkResult: resampling_rate must be at least 1 (" << resampling_rate
        << ").";
    throw std::runtime_error(msg.str());
  }
  for (int ipt = 0; ipt < number_of_points(); ++ipt) {
    const PointSample &point = points_[ipt];
    if (point.group < 0 || point.group >= number_of_groups) {
      std::ostringstream msg;
      msg << "FkResult: point " << ipt << " has group " << point.group
          << " outside [0, " << number_of_groups << ").";
      throw std::runtime_error(msg.str());
    }
    if (point.medium != specfem::plugin::Medium::acoustic &&
        point.medium != specfem::plugin::Medium::elastic) {
      std::ostringstream msg;
      msg << "FkResult: point " << ipt
          << " has a medium other than acoustic or elastic.";
      throw std::runtime_error(msg.str());
    }
  }
}

std::size_t fk_injection::FkResult::coefficient_index(int group, int series,
                                                      int coefficient) const {
  return (static_cast<std::size_t>(group) *
              fk_injection::impl::number_of_series +
          series) *
             coefficient_count_ +
         coefficient;
}

fk_injection::type_real
fk_injection::FkResult::series_coefficient(int group, int series,
                                           int coefficient) const {
  return series_coefficients_[coefficient_index(group, series, coefficient)];
}

void fk_injection::FkResult::sample(int first_step, int number_of_steps,
                                    std::span<type_real> out) const {
  const int point_count = number_of_points();
  const std::size_t expected_size = static_cast<std::size_t>(point_count) *
                                    number_of_sample_components *
                                    static_cast<std::size_t>(number_of_steps);
  if (number_of_steps < 0 || out.size() != expected_size) {
    std::ostringstream msg;
    msg << "FkResult::sample: out has " << out.size() << " values for "
        << point_count << " points, " << number_of_sample_components
        << " components and " << number_of_steps << " steps.";
    throw std::runtime_error(msg.str());
  }

  namespace cubic = fk_injection::cubic_bspline;

  const type_real *coefficients = series_coefficients_.data();
  type_real *output = out.data();
  const int coefficient_count = coefficient_count_;
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
      const PointSample &point = points_[ipt];
      const bool is_acoustic =
          point.medium == specfem::plugin::Medium::acoustic;
      const double abscissa =
          static_cast<double>(first_step + step) / resampling + point.offset;
      type_real *point_out = output + ipt * stride + step;

      const double horizontal = cubic::evaluate<type_real>(
          coefficients +
              coefficient_index(point.group,
                                fk_injection::impl::series_horizontal, 0),
          coefficient_count, abscissa);
      const double vertical = cubic::evaluate<type_real>(
          coefficients + coefficient_index(point.group,
                                           fk_injection::impl::series_vertical,
                                           0),
          coefficient_count, abscissa);

      point_out[(sample_kinematic_offset + 0) * number_of_steps] =
          static_cast<type_real>(cos_phi * horizontal);
      point_out[(sample_kinematic_offset + 1) * number_of_steps] =
          static_cast<type_real>(sin_phi * horizontal);
      point_out[(sample_kinematic_offset + 2) * number_of_steps] =
          static_cast<type_real>(vertical);

      double tx = 0.0;
      double ty = 0.0;
      double tz = 0.0;
      if (!is_acoustic) {
        const double sigma_zz = cubic::evaluate<type_real>(
            coefficients +
                coefficient_index(point.group,
                                  fk_injection::impl::series_sigma_zz, 0),
            coefficient_count, abscissa);
        const double sigma_rz = cubic::evaluate<type_real>(
            coefficients +
                coefficient_index(point.group,
                                  fk_injection::impl::series_sigma_rz, 0),
            coefficient_count, abscissa);
        const double dx = cubic::evaluate<type_real>(
            coefficients + coefficient_index(point.group,
                                             fk_injection::impl::series_dx, 0),
            coefficient_count, abscissa);
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
          static_cast<type_real>(tx);
      point_out[(sample_traction_offset + 1) * number_of_steps] =
          static_cast<type_real>(ty);
      point_out[(sample_traction_offset + 2) * number_of_steps] =
          static_cast<type_real>(tz);

      double chi_dot = 0.0;
      if (is_acoustic) {
        chi_dot = cubic::evaluate<type_real>(
            coefficients +
                coefficient_index(point.group,
                                  fk_injection::impl::series_potential_rate, 0),
            coefficient_count, abscissa);
      }
      point_out[sample_potential_rate_offset * number_of_steps] =
          static_cast<type_real>(chi_dot);
    }
  }
}
