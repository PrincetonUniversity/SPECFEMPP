#include "specfem/assembly/attenuation/dim3/attenuation.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/utilities.hpp"
#include <type_traits>

void specfem::assembly::Attenuation<specfem::element::dimension_tag::dim3>::
    init_memory_variables(
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh,
        const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
            &materials,
        const specfem::units::Hertz fc, const specfem::units::Hertz f0,
        const specfem::utilities::Band<specfem::units::Hertz> &band,
        const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
            &tau_sigma,
        const bool property_io_enabled) {

  specfem::tag_dispatch::for_each(
      attenuation_medium_combinations, [&]<typename TagsType>() {
        auto elements = element_types.get_elements_on_host(
            TagsType::medium_tag, TagsType::property_tag,
            TagsType::attenuation_tag);
        attenuation_storage.template get<TagsType>() = {
          elements, mesh,      materials,          ngllz, nglly, ngllx, fc, f0,
          band,     tau_sigma, property_io_enabled
        };
      });
}

specfem::assembly::Attenuation<specfem::element::dimension_tag::dim3>::
    Attenuation(
        const specfem::mesh::attenuation_config &config, const type_real deltat,
        const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
            &mesh,
        const specfem::assembly::element_types<
            specfem::element::dimension_tag::dim3> &element_types,
        const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
            &materials,
        const bool property_io_enabled)
    : ngllz(mesh.element_grid.ngllz), nglly(mesh.element_grid.nglly),
      ngllx(mesh.element_grid.ngllx), nspec(mesh.nspec), f0(config.f0),
      deltat(deltat) {

  if (!config.enabled)
    return;

  using specfem::units::unit_symbols::Hz;

  // Compute the band center frequency for attenuation scaling (logarithmic
  // center)
  auto fc = specfem::utilities::logarithmic_center(config.band.min.raw(),
                                                   config.band.max.raw()) *
            Hz;

  // Retain frequency-band inputs for read-time recomputation of attenuation
  // state from an on-disk Q model.
  this->fc = fc;
  this->band = config.band;
  this->tau_sigma = config.tau_sigma;

  // Compute Runge-Kutta memory-variable update coefficients from the
  // precomputed tau_sigma stored in the mesh attenuation_config.
  auto rk = specfem::attenuation::compute_integration_factors<N_SLS>(
      config.tau_sigma, deltat);

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

  init_memory_variables(element_types, mesh, materials, fc, config.f0,
                        config.band, config.tau_sigma, property_io_enabled);
}
