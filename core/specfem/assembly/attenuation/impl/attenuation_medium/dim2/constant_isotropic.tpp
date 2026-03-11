#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh/dim2/materials/materials.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/logarithmic_center.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

template <specfem::element::property_tag PropertyTag>
struct attenuation_medium<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          PropertyTag,
                          specfem::element::attenuation_tag::constant_isotropic> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim2>;

  using view_type = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  constexpr static int N_SLS = specfem::constants::N_SLS;


  // Host-only per-element scale factors
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_kappa_scale;
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_mu_scale;

  // Views: shape [nspec_attn][ngllz][ngllx][N_SLS]
  view_type kappa_relaxation_rate;
  view_type::HostMirror h_kappa_relaxation_rate;
  view_type mu_relaxation_rate;
  view_type::HostMirror h_mu_relaxation_rate;
  view_type memory_variable_kappa;
  view_type::HostMirror h_memory_variable_kappa;
  view_type memory_variable_Rxx;
  view_type::HostMirror h_memory_variable_Rxx;
  view_type memory_variable_Rxz;
  view_type::HostMirror h_memory_variable_Rxz;

  attenuation_medium() = default;

  attenuation_medium(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim2>
          &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim2>
          &materials,
      const int ngllz, const int ngllx, const type_real fc, const type_real f0,
      const specfem::utilities::FrequencyBand &band,
      const Kokkos::View<type_real [N_SLS], Kokkos::DefaultHostExecutionSpace> &tau_sigma) {

    const int nspec_attn = elements.extent(0);

    // 1. Allocate all views
    h_kappa_scale =
        Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>(
            "kappa_scale", nspec_attn);
    h_mu_scale =
        Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>(
            "mu_scale", nspec_attn);
    kappa_relaxation_rate =
        view_type("kappa_rr", nspec_attn, ngllz, ngllx, N_SLS);
    h_kappa_relaxation_rate =
        specfem::kokkos::create_mirror_view(kappa_relaxation_rate);
    mu_relaxation_rate =
        view_type("mu_rr", nspec_attn, ngllz, ngllx, N_SLS);
    h_mu_relaxation_rate =
        specfem::kokkos::create_mirror_view(mu_relaxation_rate);
    memory_variable_kappa =
        view_type("mem_kappa", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_kappa =
        specfem::kokkos::create_mirror_view(memory_variable_kappa);
    memory_variable_Rxx =
        view_type("mem_Rxx", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_Rxx =
        specfem::kokkos::create_mirror_view(memory_variable_Rxx);
    memory_variable_Rxz =
        view_type("mem_Rxz", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_Rxz =
        specfem::kokkos::create_mirror_view(memory_variable_Rxz);

    // Sync zero-initialized device views to host mirrors
    copy_to_host();

    if (nspec_attn == 0) {
      return;
    }

    // 3. Loop over elements
    for (int i = 0; i < nspec_attn; ++i) {
      const int ispec = elements(i);
      const int mesh_ispec = mesh.compute_to_mesh(ispec);

      auto material = materials.template get_material<
          specfem::element::medium_tag::elastic_psv, PropertyTag,
          specfem::element::attenuation_tag::constant_isotropic>(mesh_ispec);

      const type_real Qkappa = material.Qkappa;
      const type_real Qmu = material.Qmu;

      auto tau_eps_kappa = specfem::attenuation::compute_tau_eps<N_SLS>(
          Qkappa, tau_sigma, band.min_frequency(), band.max_frequency());
      auto tau_eps_mu = specfem::attenuation::compute_tau_eps<N_SLS>(
          Qmu, tau_sigma, band.min_frequency(), band.max_frequency());

      auto prop_kappa =
          specfem::attenuation::get_attenuation_property_values<N_SLS>(
              tau_sigma, tau_eps_kappa);
      auto prop_mu =
          specfem::attenuation::get_attenuation_property_values<N_SLS>(
              tau_sigma, tau_eps_mu);

      const type_real scale_kappa =
          specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
              fc, tau_eps_kappa, tau_sigma, Qkappa, f0);
      const type_real scale_mu =
          specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
              fc, tau_eps_mu, tau_sigma, Qmu, f0);

      h_kappa_scale(i) = scale_kappa;
      h_mu_scale(i) = scale_mu;

      // Per-GLL fill
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          for (int j = 0; j < N_SLS; ++j) {
            const type_real tauinv_j = 1.0 / tau_sigma(j);
            h_kappa_relaxation_rate(i, iz, ix, j) =
                prop_kappa.beta(j)
                * tauinv_j / prop_kappa.one_minus_sum_beta;
            h_mu_relaxation_rate(i, iz, ix, j) =
                2.0 * prop_mu.beta(j)
                * tauinv_j / prop_mu.one_minus_sum_beta;
          }
        }
      }
    }

    // 4. Push all host data (kappa/mu_cf filled; memory variables zero) to device
    copy_to_device();
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_kappa_relaxation_rate, kappa_relaxation_rate);
    Kokkos::deep_copy(h_mu_relaxation_rate, mu_relaxation_rate);
    Kokkos::deep_copy(h_memory_variable_kappa, memory_variable_kappa);
    Kokkos::deep_copy(h_memory_variable_Rxx, memory_variable_Rxx);
    Kokkos::deep_copy(h_memory_variable_Rxz, memory_variable_Rxz);
  }

  void copy_to_device() {
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
    Kokkos::deep_copy(memory_variable_kappa, h_memory_variable_kappa);
    Kokkos::deep_copy(memory_variable_Rxx, h_memory_variable_Rxx);
    Kokkos::deep_copy(memory_variable_Rxz, h_memory_variable_Rxz);
  }
};

} // namespace specfem::assembly::impl
