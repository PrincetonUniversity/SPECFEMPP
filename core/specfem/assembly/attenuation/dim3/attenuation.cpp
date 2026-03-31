#include "specfem/assembly/attenuation/dim3/attenuation.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/utilities.hpp"
#include <type_traits>

void specfem::assembly::Attenuation<
    specfem::element::dimension_tag::dim3,
    specfem::element::attenuation_tag::constant_isotropic>::
    init_memory_variables(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh,
        const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
            &materials,
        const type_real fc, const type_real f0,
        const specfem::utilities::Band<specfem::units::Hertz> &band,
        const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
            &tau_sigma){

      FOR_EACH_IN_PRODUCT(
          (DIMENSION_TAG(DIM3), MEDIUM_TAG(ELASTIC), PROPERTY_TAG(ISOTROPIC),
           ATTENUATION_TAG(CONSTANT_ISOTROPIC)),
          CAPTURE(attn_medium) {
            using medium_t = std::remove_reference_t<decltype(_attn_medium_)>;
            auto elements = element_types.get_elements_on_host(
                _medium_tag_, _property_tag_, _attenuation_tag_);
            _attn_medium_ = medium_t(elements, mesh, materials, ngllz, nglly,
                                     ngllx, fc, f0, band, tau_sigma);
          })
    }

specfem::assembly::Attenuation<
    specfem::element::dimension_tag::dim3,
    specfem::element::attenuation_tag::constant_isotropic>::
    Attenuation(
        const specfem::units::Hertz reference_frequency,
        const specfem::units::Hertz min_frequency,
        const specfem::units::Hertz max_frequency,
        const bool auto_compute_attenuation_band, const type_real deltat,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh,
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::assembly::Info<specfem::element::dimension_tag::dim3>
            &info,
        const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
            &materials)
    : ngllz(mesh.element_grid.ngllz), nglly(mesh.element_grid.nglly),
      ngllx(mesh.element_grid.ngllx), nspec(mesh.nspec),
      f0(reference_frequency),
      auto_compute_attenuation_band(auto_compute_attenuation_band) {

  specfem::utilities::Band<specfem::units::Hertz> band =
      auto_compute_attenuation_band
          ? attenuation::compute_band<N_SLS>(
                specfem::units::Seconds(info.largest_minimum_period))
          : specfem::utilities::Band<specfem::units::Hertz>(min_frequency,
                                                            max_frequency);

  // Compute the band center frequency for attenuation scaling (logarithmic
  // center)
  type_real fc =
      specfem::utilities::logarithmic_center(band.min.raw(), band.max.raw());

  // Compute tau_sigma once (shared across all elements)
  auto tau_sigma = specfem::attenuation::compute_tau_sigma<N_SLS>(
      band.min.raw(), band.max.raw());

  // Compute Runge-Kutta memory-variable update coefficients
  auto rk = specfem::attenuation::compute_integration_factors<N_SLS>(tau_sigma,
                                                                     deltat);
  this->alpha_rk = rk.alpha;
  this->beta_rk = rk.beta;
  this->gamma_rk = rk.gamma;

  // Copy RK coefficients to device for use in device kernels
  this->d_alpha_rk = Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>("d_alpha_rk");
  Kokkos::deep_copy(this->d_alpha_rk, this->alpha_rk);
  this->d_beta_rk = Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                 Kokkos::DefaultExecutionSpace>("d_beta_rk");
  Kokkos::deep_copy(this->d_beta_rk, this->beta_rk);
  this->d_gamma_rk = Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>("d_gamma_rk");
  Kokkos::deep_copy(this->d_gamma_rk, this->gamma_rk);

  init_memory_variables(element_types, mesh, materials, fc, f0.raw(), band,
                        tau_sigma);
}
