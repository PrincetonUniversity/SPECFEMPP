#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh/dim3/materials/materials.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

template <specfem::element::property_tag PropertyTag>
struct attenuation_medium<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          PropertyTag,
                          specfem::element::attenuation_tag::constant_isotropic> :
                          specfem::data_access::Container<
                              specfem::data_access::ContainerType::domain,
                              specfem::data_access::DataClassType::attenuation,
                              specfem::element::dimension_tag::dim3> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim3>;

  using view_type = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  constexpr static int N_SLS = specfem::constants::N_SLS;

  // Host-only per-element scale factors
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_kappa_scale;
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_mu_scale;

  // Views: shape [nspec_attn][ngllz][nglly][ngllx][N_SLS]
  view_type kappa_relaxation_rate;
  view_type::HostMirror h_kappa_relaxation_rate;
  view_type mu_relaxation_rate;
  view_type::HostMirror h_mu_relaxation_rate;
  view_type memory_variable_kappa;
  view_type::HostMirror h_memory_variable_kappa;
  view_type memory_variable_Rxx;
  view_type::HostMirror h_memory_variable_Rxx;
  view_type memory_variable_Ryy;
  view_type::HostMirror h_memory_variable_Ryy;
  view_type memory_variable_Rzz;
  view_type::HostMirror h_memory_variable_Rzz;
  view_type memory_variable_Rxy;
  view_type::HostMirror h_memory_variable_Rxy;
  view_type memory_variable_Rxz;
  view_type::HostMirror h_memory_variable_Rxz;
  view_type memory_variable_Ryz;
  view_type::HostMirror h_memory_variable_Ryz;

  // Index mapping: global ispec -> compact attenuation index (-1 if not attenuating)
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
      h_attenuation_index_mapping;
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
      attenuation_index_mapping;

  attenuation_medium() = default;

  attenuation_medium(
      const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> elements,
      const specfem::assembly::mesh<
          specfem::element::dimension_tag::dim3> &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
          &materials,
      const int ngllz, const int nglly, const int ngllx,
      const type_real fc, const type_real f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
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
        view_type("kappa_relaxation_rate", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_kappa_relaxation_rate =
        Kokkos::create_mirror_view(kappa_relaxation_rate);
    mu_relaxation_rate =
        view_type("mu_relaxation_rate", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_mu_relaxation_rate =
        Kokkos::create_mirror_view(mu_relaxation_rate);
    memory_variable_kappa =
        view_type("mem_kappa", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_kappa =
        Kokkos::create_mirror_view(memory_variable_kappa);
    memory_variable_Rxx =
        view_type("mem_Rxx", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxx =
        Kokkos::create_mirror_view(memory_variable_Rxx);
    memory_variable_Ryy =
        view_type("mem_Ryy", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Ryy =
        Kokkos::create_mirror_view(memory_variable_Ryy);
    memory_variable_Rzz =
        view_type("mem_Rzz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rzz =
        Kokkos::create_mirror_view(memory_variable_Rzz);
    memory_variable_Rxy =
        view_type("mem_Rxy", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxy =
        Kokkos::create_mirror_view(memory_variable_Rxy);
    memory_variable_Rxz =
        view_type("mem_Rxz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxz =
        Kokkos::create_mirror_view(memory_variable_Rxz);
    memory_variable_Ryz =
        view_type("mem_Ryz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Ryz =
        Kokkos::create_mirror_view(memory_variable_Ryz);

    // Allocate and populate the inverse index mapping (global ispec -> compact index)
    h_attenuation_index_mapping =
        Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>(
            "h_attenuation_index_mapping", mesh.nspec);
    Kokkos::deep_copy(h_attenuation_index_mapping, -1);
    for (int i = 0; i < nspec_attn; ++i) {
      h_attenuation_index_mapping(elements(i)) = i;
    }
    attenuation_index_mapping =
        Kokkos::View<int *, Kokkos::DefaultExecutionSpace>(
            "attenuation_index_mapping", mesh.nspec);
    Kokkos::deep_copy(attenuation_index_mapping, h_attenuation_index_mapping);

    // Sync zero-initialized device views to host mirrors
    copy_to_host();

    if (nspec_attn == 0) {
      return;
    }

    // 3. Loop over elements
    for (int i = 0; i < nspec_attn; ++i) {
      const int ispec = elements(i);

      auto material = materials.template get_material<
          specfem::element::medium_tag::elastic, PropertyTag,
          specfem::element::attenuation_tag::constant_isotropic>(ispec);

      material.compute_attenuation_properties(fc, f0, band, tau_sigma);

      // Per-GLL fill
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            for (int j = 0; j < N_SLS; ++j) {

              const type_real tauinv_j = 1.0 / tau_sigma(j);

              auto kappa_props = material.get_kappa_attenuation_properties();

              h_kappa_relaxation_rate(i, iz, iy, ix, j) =
                  kappa_props.beta(j)
                  * tauinv_j / kappa_props.one_minus_sum_beta;

              auto mu_props = material.get_mu_attenuation_properties();

              h_mu_relaxation_rate(i, iz, iy, ix, j) =
                  2.0 * mu_props.beta(j)
                  * tauinv_j / mu_props.one_minus_sum_beta;
            }
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
    Kokkos::deep_copy(h_memory_variable_Ryy, memory_variable_Ryy);
    Kokkos::deep_copy(h_memory_variable_Rzz, memory_variable_Rzz);
    Kokkos::deep_copy(h_memory_variable_Rxy, memory_variable_Rxy);
    Kokkos::deep_copy(h_memory_variable_Rxz, memory_variable_Rxz);
    Kokkos::deep_copy(h_memory_variable_Ryz, memory_variable_Ryz);
  }

  void copy_to_device() {
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
    Kokkos::deep_copy(memory_variable_kappa, h_memory_variable_kappa);
    Kokkos::deep_copy(memory_variable_Rxx, h_memory_variable_Rxx);
    Kokkos::deep_copy(memory_variable_Ryy, h_memory_variable_Ryy);
    Kokkos::deep_copy(memory_variable_Rzz, h_memory_variable_Rzz);
    Kokkos::deep_copy(memory_variable_Rxy, h_memory_variable_Rxy);
    Kokkos::deep_copy(memory_variable_Rxz, h_memory_variable_Rxz);
    Kokkos::deep_copy(memory_variable_Ryz, h_memory_variable_Ryz);
  }

  /**
   * @brief Load attenuation data for a single GLL point from device views
   *        into a point-local attenuation struct.
   *
   * Populates relaxation rates and memory variables. The global RK
   * coefficients are NOT populated here; they are added by the outer
   * load_on_device free function.
   */
  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void load_device_values(const IndexType &index,
                                                 PointType &point) const {
    const int i = attenuation_index_mapping(index.ispec);
    if constexpr (!IndexType::using_simd) {
      for (int j = 0; j < N_SLS; ++j) {
        point.kappa_relaxation_rate(j) =
            kappa_relaxation_rate(i, index.iz, index.iy, index.ix, j);
        point.mu_relaxation_rate(j) =
            mu_relaxation_rate(i, index.iz, index.iy, index.ix, j);
        point.Rxx(j) = memory_variable_Rxx(i, index.iz, index.iy, index.ix, j);
        point.Ryy(j) = memory_variable_Ryy(i, index.iz, index.iy, index.ix, j);
        point.Rxy(j) = memory_variable_Rxy(i, index.iz, index.iy, index.ix, j);
        point.Rxz(j) = memory_variable_Rxz(i, index.iz, index.iy, index.ix, j);
        point.Ryz(j) = memory_variable_Ryz(i, index.iz, index.iy, index.ix, j);
        point.Rkappa(j) =
            memory_variable_kappa(i, index.iz, index.iy, index.ix, j);
      }
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      mask_type mask([&](std::size_t lane) { return index.mask(lane); });
      for (int j = 0; j < N_SLS; ++j) {
        Kokkos::Experimental::where(mask, point.kappa_relaxation_rate(j))
            .copy_from(
                &kappa_relaxation_rate(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.mu_relaxation_rate(j))
            .copy_from(
                &mu_relaxation_rate(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Rxx(j))
            .copy_from(
                &memory_variable_Rxx(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Ryy(j))
            .copy_from(
                &memory_variable_Ryy(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Rxy(j))
            .copy_from(
                &memory_variable_Rxy(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Rxz(j))
            .copy_from(
                &memory_variable_Rxz(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Ryz(j))
            .copy_from(
                &memory_variable_Ryz(i, index.iz, index.iy, index.ix, j),
                tag_type());
        Kokkos::Experimental::where(mask, point.Rkappa(j))
            .copy_from(
                &memory_variable_kappa(i, index.iz, index.iy, index.ix, j),
                tag_type());
      }
    }
  }

  /**
   * @brief Store evolved SLS memory variables from a point-local struct back
   *        to the device views.
   *
   * Only the memory variables are written; relaxation rates are
   * simulation-lifetime constants and are not written back.
   * Note: memory_variable_Rzz = -(Rxx + Ryy) and is not stored in the point
   * type; callers must update it separately if needed.
   */
  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointType &point) const {
    const int i = attenuation_index_mapping(index.ispec);
    if constexpr (!IndexType::using_simd) {
      for (int j = 0; j < N_SLS; ++j) {
        memory_variable_Rxx(i, index.iz, index.iy, index.ix, j) = point.Rxx(j);
        memory_variable_Ryy(i, index.iz, index.iy, index.ix, j) = point.Ryy(j);
        memory_variable_Rxy(i, index.iz, index.iy, index.ix, j) = point.Rxy(j);
        memory_variable_Rxz(i, index.iz, index.iy, index.ix, j) = point.Rxz(j);
        memory_variable_Ryz(i, index.iz, index.iy, index.ix, j) = point.Ryz(j);
        memory_variable_kappa(i, index.iz, index.iy, index.ix, j) =
            point.Rkappa(j);
      }
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      mask_type mask([&](std::size_t lane) { return index.mask(lane); });
      for (int j = 0; j < N_SLS; ++j) {
        Kokkos::Experimental::where(mask, point.Rxx(j))
            .copy_to(&memory_variable_Rxx(i, index.iz, index.iy, index.ix, j),
                     tag_type());
        Kokkos::Experimental::where(mask, point.Ryy(j))
            .copy_to(&memory_variable_Ryy(i, index.iz, index.iy, index.ix, j),
                     tag_type());
        Kokkos::Experimental::where(mask, point.Rxy(j))
            .copy_to(&memory_variable_Rxy(i, index.iz, index.iy, index.ix, j),
                     tag_type());
        Kokkos::Experimental::where(mask, point.Rxz(j))
            .copy_to(&memory_variable_Rxz(i, index.iz, index.iy, index.ix, j),
                     tag_type());
        Kokkos::Experimental::where(mask, point.Ryz(j))
            .copy_to(&memory_variable_Ryz(i, index.iz, index.iy, index.ix, j),
                     tag_type());
        Kokkos::Experimental::where(mask, point.Rkappa(j))
            .copy_to(
                &memory_variable_kappa(i, index.iz, index.iy, index.ix, j),
                tag_type());
      }
    }
  }
};

} // namespace specfem::assembly::impl
