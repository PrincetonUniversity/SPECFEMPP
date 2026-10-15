#include "specfem/injection/fk/fk_result.hpp"

#include "specfem/injection/fk/series_layout.hpp" // series slot indices
#include "specfem/injection/frame_layout.hpp"     // frame-buffer components
#include <Kokkos_Core.hpp>
#include <type_traits>

specfem::injection::fk::FkResult::FkResult(
    Kokkos::View<type_real ***, Kokkos::LayoutRight> series_coefficients,
    Kokkos::View<int *> point_group, Kokkos::View<double *> point_offset,
    Kokkos::View<type_real **, Kokkos::LayoutRight> point_material,
    Kokkos::View<int *> point_is_acoustic, type_real cos_phi, type_real sin_phi,
    int resampling_rate, bool has_traction)
    : series_coefficients_(std::move(series_coefficients)),
      point_group_(std::move(point_group)),
      point_offset_(std::move(point_offset)),
      point_material_(std::move(point_material)),
      point_is_acoustic_(std::move(point_is_acoustic)), cos_phi_(cos_phi),
      sin_phi_(sin_phi), resampling_rate_(resampling_rate),
      number_of_points_(static_cast<int>(point_group_.extent(0))),
      number_of_groups_(static_cast<int>(series_coefficients_.extent(0))),
      coefficient_count_(static_cast<int>(series_coefficients_.extent(2))),
      has_traction_(has_traction) {}

void specfem::injection::fk::FkResult::sample(
    int first_step, int number_of_steps,
    const Kokkos::View<type_real ***> &out) const {
  static_assert(
      std::is_same<
          Kokkos::View<type_real ***, Kokkos::LayoutRight>::array_layout,
          Kokkos::LayoutRight>::value,
      "series coefficients must be LayoutRight for contiguous spline "
      "evaluation");

  namespace cubic = specfem::utilities::cubic_bspline;
  namespace frame = specfem::injection::frame_component;
  constexpr int s0 = specfem::injection::fk_impl::series_horizontal;
  constexpr int s1 = specfem::injection::fk_impl::series_vertical;
  constexpr int s2 = specfem::injection::fk_impl::series_sigma_zz;
  constexpr int s3 = specfem::injection::fk_impl::series_sigma_rz;
  constexpr int s4 = specfem::injection::fk_impl::series_dx;
  constexpr int with_traction =
      specfem::injection::displacement_traction_component_count;
  constexpr int with_potential =
      specfem::injection::displacement_traction_potential_component_count;

  // Local copies (a device lambda cannot capture `this`).
  const auto coefficients = series_coefficients_;
  const auto group = point_group_;
  const auto offset = point_offset_;
  const auto material = point_material_;
  const auto is_acoustic = point_is_acoustic_;
  const int ncoef = coefficient_count_;
  const int resampling = resampling_rate_;
  const double cos_phi = static_cast<double>(cos_phi_);
  const double sin_phi = static_cast<double>(sin_phi_);
  const int ncomp = static_cast<int>(out.extent(1));
  const int steps = number_of_steps;

  const int total = number_of_points_ * number_of_steps;
  Kokkos::parallel_for(
      "fk_sample", Kokkos::RangePolicy<>(0, total), KOKKOS_LAMBDA(int idx) {
        const int k = idx % steps;
        const int ipt = idx / steps;
        const int g = group(ipt);
        const double abscissa = static_cast<double>(first_step + k) /
                                    static_cast<double>(resampling) +
                                offset(ipt);

        const double horizontal = cubic::evaluate<type_real>(
            &coefficients(g, s0, 0), ncoef, abscissa);
        const double vertical = cubic::evaluate<type_real>(
            &coefficients(g, s1, 0), ncoef, abscissa);

        out(ipt, frame::displacement_x, k) =
            static_cast<type_real>(cos_phi * horizontal);
        out(ipt, frame::displacement_y, k) =
            static_cast<type_real>(sin_phi * horizontal);
        out(ipt, frame::displacement_z, k) = static_cast<type_real>(vertical);

        if (ncomp >= with_traction) {
          double tx = 0.0;
          double ty = 0.0;
          double tz = 0.0;
          if (!is_acoustic(ipt)) {
            const double sigma_zz = cubic::evaluate<type_real>(
                &coefficients(g, s2, 0), ncoef, abscissa);
            const double sigma_rz = cubic::evaluate<type_real>(
                &coefficients(g, s3, 0), ncoef, abscissa);
            const double dx = cubic::evaluate<type_real>(
                &coefficients(g, s4, 0), ncoef, abscissa);
            const double xi1 = static_cast<double>(material(ipt, 0));
            const double xim = static_cast<double>(material(ipt, 1));
            const double bulk = static_cast<double>(material(ipt, 2));
            const double nx = static_cast<double>(material(ipt, 3));
            const double ny = static_cast<double>(material(ipt, 4));
            const double nz = static_cast<double>(material(ipt, 5));

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
          out(ipt, frame::traction_x, k) = static_cast<type_real>(tx);
          out(ipt, frame::traction_y, k) = static_cast<type_real>(ty);
          out(ipt, frame::traction_z, k) = static_cast<type_real>(tz);
        }

        if (ncomp >= with_potential) {
          // Acoustic points carry the incident potential-rate χ̇ (series S2);
          // zero for elastic points (no fluid potential).
          double chi_dot = 0.0;
          if (is_acoustic(ipt)) {
            chi_dot = cubic::evaluate<type_real>(&coefficients(g, s2, 0), ncoef,
                                                 abscissa);
          }
          out(ipt, frame::potential_rate, k) = static_cast<type_real>(chi_dot);
        }
      });
  Kokkos::fence();
}
