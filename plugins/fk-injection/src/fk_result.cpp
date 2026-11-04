#include "fk_injection/fk_result.hpp"

#include "impl/series_layout.hpp" // series slot indices
#include <cstddef>
#include <utility>

fk_injection::FkResult::FkResult(std::vector<type_real> series_coefficients,
                                 int number_of_groups, int coefficient_count,
                                 std::vector<int> point_group,
                                 std::vector<double> point_offset,
                                 std::vector<type_real> point_material,
                                 std::vector<int> point_is_acoustic,
                                 type_real cos_phi, type_real sin_phi,
                                 int resampling_rate, bool has_traction)
    : series_coefficients_(std::move(series_coefficients)),
      point_group_(std::move(point_group)),
      point_offset_(std::move(point_offset)),
      point_material_(std::move(point_material)),
      point_is_acoustic_(std::move(point_is_acoustic)), cos_phi_(cos_phi),
      sin_phi_(sin_phi), resampling_rate_(resampling_rate),
      number_of_points_(static_cast<int>(point_group_.size())),
      number_of_groups_(number_of_groups),
      coefficient_count_(coefficient_count), has_traction_(has_traction) {}

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
                                    int number_of_components,
                                    type_real *out) const {
  namespace cubic = fk_injection::cubic_bspline;
  constexpr int s0 = fk_injection::impl::series_horizontal;
  constexpr int s1 = fk_injection::impl::series_vertical;
  constexpr int s2 = fk_injection::impl::series_sigma_zz;
  constexpr int s3 = fk_injection::impl::series_sigma_rz;
  constexpr int s4 = fk_injection::impl::series_dx;

  const type_real *coefficients = series_coefficients_.data();
  const type_real *material = point_material_.data();
  const int ncoef = coefficient_count_;
  const int resampling = resampling_rate_;
  const double cos_phi = static_cast<double>(cos_phi_);
  const double sin_phi = static_cast<double>(sin_phi_);
  const int ncomp = number_of_components;
  const int steps = number_of_steps;
  const std::size_t stride = static_cast<std::size_t>(ncomp) * steps;

#ifdef FK_INJECTION_ENABLE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int ipt = 0; ipt < number_of_points_; ++ipt) {
    for (int k = 0; k < steps; ++k) {
      const int g = point_group_[ipt];
      const double abscissa = static_cast<double>(first_step + k) /
                                  static_cast<double>(resampling) +
                              point_offset_[ipt];
      type_real *point_out = out + ipt * stride + k;
      const type_real *point_material = material + ipt * std::size_t{ 6 };

      const double horizontal = cubic::evaluate<type_real>(
          coefficients + coefficient_index(g, s0, 0), ncoef, abscissa);
      const double vertical = cubic::evaluate<type_real>(
          coefficients + coefficient_index(g, s1, 0), ncoef, abscissa);

      point_out[0 * steps] = static_cast<type_real>(cos_phi * horizontal);
      point_out[1 * steps] = static_cast<type_real>(sin_phi * horizontal);
      point_out[2 * steps] = static_cast<type_real>(vertical);

      if (ncomp > 3) {
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        if (!point_is_acoustic_[ipt]) {
          const double sigma_zz = cubic::evaluate<type_real>(
              coefficients + coefficient_index(g, s2, 0), ncoef, abscissa);
          const double sigma_rz = cubic::evaluate<type_real>(
              coefficients + coefficient_index(g, s3, 0), ncoef, abscissa);
          const double dx = cubic::evaluate<type_real>(
              coefficients + coefficient_index(g, s4, 0), ncoef, abscissa);
          const double xi1 = static_cast<double>(point_material[0]);
          const double xim = static_cast<double>(point_material[1]);
          const double bulk = static_cast<double>(point_material[2]);
          const double nx = static_cast<double>(point_material[3]);
          const double ny = static_cast<double>(point_material[4]);
          const double nz = static_cast<double>(point_material[5]);

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
        point_out[3 * steps] = static_cast<type_real>(tx);
        point_out[4 * steps] = static_cast<type_real>(ty);
        point_out[5 * steps] = static_cast<type_real>(tz);
      }

      if (ncomp > 6) {
        double pressure = 0.0;
        if (point_is_acoustic_[ipt]) {
          pressure = cubic::evaluate<type_real>(
              coefficients + coefficient_index(g, s2, 0), ncoef, abscissa);
        }
        point_out[6 * steps] = static_cast<type_real>(pressure);
      }
    }
  }
}
