#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/datatype.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh/dim3/materials/materials.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::impl {

template <specfem::element::property_tag PropertyTag>
struct attenuation_medium<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic, PropertyTag,
                          specfem::element::attenuation_tag::constant_isotropic>
    : specfem::data_access::Container<
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

  // Scalar-per-GLL view: shape [nspec_attn][ngllz][nglly][ngllx]
  using scalar_view_type = typename base_type::template scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  // Host-only per-element scale factors
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_kappa_scale;
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_mu_scale;

  // Views: shape [nspec_attn][ngllz][nglly][ngllx][N_SLS]
  view_type kappa_relaxation_rate;
  view_type::host_mirror_type h_kappa_relaxation_rate;
  view_type mu_relaxation_rate;
  view_type::host_mirror_type h_mu_relaxation_rate;
  view_type memory_variable_kappa;
  view_type::host_mirror_type h_memory_variable_kappa;
  view_type memory_variable_Rxx;
  view_type::host_mirror_type h_memory_variable_Rxx;
  view_type memory_variable_Ryy;
  view_type::host_mirror_type h_memory_variable_Ryy;
  view_type memory_variable_Rzz;
  view_type::host_mirror_type h_memory_variable_Rzz;
  view_type memory_variable_Rxy;
  view_type::host_mirror_type h_memory_variable_Rxy;
  view_type memory_variable_Rxz;
  view_type::host_mirror_type h_memory_variable_Rxz;
  view_type memory_variable_Ryz;
  view_type::host_mirror_type h_memory_variable_Ryz;

  // Symmetrised strain components from previous Taylor step: shape
  // [nspec_attn][ngllz][nglly][ngllx]
  scalar_view_type epsilon_xx_att;
  scalar_view_type::host_mirror_type h_epsilon_xx_att;
  scalar_view_type epsilon_yy_att;
  scalar_view_type::host_mirror_type h_epsilon_yy_att;
  scalar_view_type epsilon_zz_att;
  scalar_view_type::host_mirror_type h_epsilon_zz_att;
  scalar_view_type epsilon_xy_att;
  scalar_view_type::host_mirror_type h_epsilon_xy_att;
  scalar_view_type epsilon_xz_att;
  scalar_view_type::host_mirror_type h_epsilon_xz_att;
  scalar_view_type epsilon_yz_att;
  scalar_view_type::host_mirror_type h_epsilon_yz_att;

  // Index mapping: global ispec -> compact attenuation index (-1 if not
  // attenuating)
  Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>
      h_attenuation_index_mapping;
  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> attenuation_index_mapping;

  attenuation_medium() = default;

  template <typename ViewType>
  attenuation_medium(
      const ViewType &elements,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
          &materials,
      const int ngllz, const int nglly, const int ngllx,
      const specfem::units::Hertz fc, const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) {

    const int nspec_attn = elements.extent(0);

    // 1. Allocate all views
    h_kappa_scale =
        Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>(
            "kappa_scale", nspec_attn);
    h_mu_scale = Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace>(
        "mu_scale", nspec_attn);
    kappa_relaxation_rate = view_type("kappa_relaxation_rate", nspec_attn,
                                      ngllz, nglly, ngllx, N_SLS);
    h_kappa_relaxation_rate =
        specfem::datatype::create_mirror_view(kappa_relaxation_rate);
    mu_relaxation_rate =
        view_type("mu_relaxation_rate", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_mu_relaxation_rate =
        specfem::datatype::create_mirror_view(mu_relaxation_rate);
    memory_variable_kappa =
        view_type("mem_kappa", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_kappa =
        specfem::datatype::create_mirror_view(memory_variable_kappa);
    memory_variable_Rxx =
        view_type("mem_Rxx", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxx =
        specfem::datatype::create_mirror_view(memory_variable_Rxx);
    memory_variable_Ryy =
        view_type("mem_Ryy", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Ryy =
        specfem::datatype::create_mirror_view(memory_variable_Ryy);
    memory_variable_Rzz =
        view_type("mem_Rzz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rzz =
        specfem::datatype::create_mirror_view(memory_variable_Rzz);
    memory_variable_Rxy =
        view_type("mem_Rxy", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxy =
        specfem::datatype::create_mirror_view(memory_variable_Rxy);
    memory_variable_Rxz =
        view_type("mem_Rxz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Rxz =
        specfem::datatype::create_mirror_view(memory_variable_Rxz);
    memory_variable_Ryz =
        view_type("mem_Ryz", nspec_attn, ngllz, nglly, ngllx, N_SLS);
    h_memory_variable_Ryz =
        specfem::datatype::create_mirror_view(memory_variable_Ryz);

    epsilon_xx_att =
        scalar_view_type("epsilon_xx_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_xx_att = specfem::datatype::create_mirror_view(epsilon_xx_att);
    Kokkos::deep_copy(epsilon_xx_att, static_cast<type_real>(0));
    epsilon_yy_att =
        scalar_view_type("epsilon_yy_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_yy_att = specfem::datatype::create_mirror_view(epsilon_yy_att);
    Kokkos::deep_copy(epsilon_yy_att, static_cast<type_real>(0));
    epsilon_zz_att =
        scalar_view_type("epsilon_zz_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_zz_att = specfem::datatype::create_mirror_view(epsilon_zz_att);
    Kokkos::deep_copy(epsilon_zz_att, static_cast<type_real>(0));
    epsilon_xy_att =
        scalar_view_type("epsilon_xy_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_xy_att = specfem::datatype::create_mirror_view(epsilon_xy_att);
    Kokkos::deep_copy(epsilon_xy_att, static_cast<type_real>(0));
    epsilon_xz_att =
        scalar_view_type("epsilon_xz_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_xz_att = specfem::datatype::create_mirror_view(epsilon_xz_att);
    Kokkos::deep_copy(epsilon_xz_att, static_cast<type_real>(0));
    epsilon_yz_att =
        scalar_view_type("epsilon_yz_att", nspec_attn, ngllz, nglly, ngllx);
    h_epsilon_yz_att = specfem::datatype::create_mirror_view(epsilon_yz_att);
    Kokkos::deep_copy(epsilon_yz_att, static_cast<type_real>(0));

    // Allocate and populate the inverse index mapping (global ispec -> compact
    // index)
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

    // Sanity check input frequencies before looping over elements
    if (fc.raw() <= 0 || f0.raw() <= 0) {
      throw std::runtime_error(
          "Center frequency fc and reference frequency f0 must be positive.");
    }

    // 3. Loop over elements
    for (int i = 0; i < nspec_attn; ++i) {
      const int ispec = elements(i);
      const int mesh_ispec = mesh.h_compute_to_mesh(ispec);

      auto material = materials.template get_material<
          specfem::element::medium_tag::elastic, PropertyTag,
          specfem::element::attenuation_tag::constant_isotropic>(mesh_ispec);

      auto computed_values = material.compute_attenuation_properties(
          f0.raw(), fc.raw(), band, tau_sigma);
      auto kappa_props = computed_values.kappa_attenuation_properties;
      auto mu_props = computed_values.mu_attenuation_properties;

      // Scaled moduli: matches SPECFEM3D's kappastore/mustore after
      // prepare_attenuation.f90 applies scale_factor in-place.
      const auto scaled_props = material.get_properties();
      const type_real kappa_sc = scaled_props.kappa();
      const type_real mu_sc = scaled_props.mu();

      // Per-GLL fill
      for (int j = 0; j < N_SLS; ++j) {

        // Compute per element relaxation rates (modulus included, matching
        // SPECFEM3D's factor_loc = modulus * factor_common)
        const type_real tauinv_j = 1.0 / tau_sigma(j);
        auto kappa_rr_j = kappa_sc * kappa_props.beta(j) * tauinv_j /
                          kappa_props.one_minus_sum_beta;
        auto mu_rr_j = mu_sc * 2.0 * mu_props.beta(j) * tauinv_j /
                       mu_props.one_minus_sum_beta;

        // Assigning relaxation rates to all GLL points
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int iy = 0; iy < nglly; ++iy) {
            for (int ix = 0; ix < ngllx; ++ix) {
              h_kappa_relaxation_rate(i, iz, iy, ix, j) = kappa_rr_j;
              h_mu_relaxation_rate(i, iz, iy, ix, j) = mu_rr_j;
            }
          }
        }
      }
    }

    // 4. Push all host data (kappa/mu_cf filled; memory variables zero) to
    // device
    copy_to_device();
  }

  void copy_to_host() {
    specfem::datatype::deep_copy(h_kappa_relaxation_rate,
                                 kappa_relaxation_rate);
    specfem::datatype::deep_copy(h_mu_relaxation_rate, mu_relaxation_rate);
    specfem::datatype::deep_copy(h_memory_variable_kappa,
                                 memory_variable_kappa);
    specfem::datatype::deep_copy(h_memory_variable_Rxx, memory_variable_Rxx);
    specfem::datatype::deep_copy(h_memory_variable_Ryy, memory_variable_Ryy);
    specfem::datatype::deep_copy(h_memory_variable_Rzz, memory_variable_Rzz);
    specfem::datatype::deep_copy(h_memory_variable_Rxy, memory_variable_Rxy);
    specfem::datatype::deep_copy(h_memory_variable_Rxz, memory_variable_Rxz);
    specfem::datatype::deep_copy(h_memory_variable_Ryz, memory_variable_Ryz);
    specfem::datatype::deep_copy(h_epsilon_xx_att, epsilon_xx_att);
    specfem::datatype::deep_copy(h_epsilon_yy_att, epsilon_yy_att);
    specfem::datatype::deep_copy(h_epsilon_zz_att, epsilon_zz_att);
    specfem::datatype::deep_copy(h_epsilon_xy_att, epsilon_xy_att);
    specfem::datatype::deep_copy(h_epsilon_xz_att, epsilon_xz_att);
    specfem::datatype::deep_copy(h_epsilon_yz_att, epsilon_yz_att);
  }

  void copy_to_device() {
    specfem::datatype::deep_copy(kappa_relaxation_rate,
                                 h_kappa_relaxation_rate);
    specfem::datatype::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
    specfem::datatype::deep_copy(memory_variable_kappa,
                                 h_memory_variable_kappa);
    specfem::datatype::deep_copy(memory_variable_Rxx, h_memory_variable_Rxx);
    specfem::datatype::deep_copy(memory_variable_Ryy, h_memory_variable_Ryy);
    specfem::datatype::deep_copy(memory_variable_Rzz, h_memory_variable_Rzz);
    specfem::datatype::deep_copy(memory_variable_Rxy, h_memory_variable_Rxy);
    specfem::datatype::deep_copy(memory_variable_Rxz, h_memory_variable_Rxz);
    specfem::datatype::deep_copy(memory_variable_Ryz, h_memory_variable_Ryz);
    specfem::datatype::deep_copy(epsilon_xx_att, h_epsilon_xx_att);
    specfem::datatype::deep_copy(epsilon_yy_att, h_epsilon_yy_att);
    specfem::datatype::deep_copy(epsilon_zz_att, h_epsilon_zz_att);
    specfem::datatype::deep_copy(epsilon_xy_att, h_epsilon_xy_att);
    specfem::datatype::deep_copy(epsilon_xz_att, h_epsilon_xz_att);
    specfem::datatype::deep_copy(epsilon_yz_att, h_epsilon_yz_att);
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
      point.epsilon_xx = epsilon_xx_att(i, index.iz, index.iy, index.ix);
      point.epsilon_yy = epsilon_yy_att(i, index.iz, index.iy, index.ix);
      point.epsilon_zz = epsilon_zz_att(i, index.iz, index.iy, index.ix);
      point.epsilon_xy = epsilon_xy_att(i, index.iz, index.iy, index.ix);
      point.epsilon_xz = epsilon_xz_att(i, index.iz, index.iy, index.ix);
      point.epsilon_yz = epsilon_yz_att(i, index.iz, index.iy, index.ix);
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      const auto mask = index.template get_mask<simd>();
      for (int j = 0; j < N_SLS; ++j) {
        point.kappa_relaxation_rate(j) =
            Kokkos::Experimental::simd_partial_load(
                &kappa_relaxation_rate(i, index.iz, index.iy, index.ix, j),
                mask, tag_type());
        point.mu_relaxation_rate(j) = Kokkos::Experimental::simd_partial_load(
            &mu_relaxation_rate(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Rxx(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Rxx(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Ryy(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Ryy(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Rxy(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Rxy(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Rxz(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Rxz(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Ryz(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Ryz(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        point.Rkappa(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_kappa(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
      }
      point.epsilon_xx = Kokkos::Experimental::simd_partial_load(
          &epsilon_xx_att(i, index.iz, index.iy, index.ix), mask, tag_type());
      point.epsilon_yy = Kokkos::Experimental::simd_partial_load(
          &epsilon_yy_att(i, index.iz, index.iy, index.ix), mask, tag_type());
      point.epsilon_zz = Kokkos::Experimental::simd_partial_load(
          &epsilon_zz_att(i, index.iz, index.iy, index.ix), mask, tag_type());
      point.epsilon_xy = Kokkos::Experimental::simd_partial_load(
          &epsilon_xy_att(i, index.iz, index.iy, index.ix), mask, tag_type());
      point.epsilon_xz = Kokkos::Experimental::simd_partial_load(
          &epsilon_xz_att(i, index.iz, index.iy, index.ix), mask, tag_type());
      point.epsilon_yz = Kokkos::Experimental::simd_partial_load(
          &epsilon_yz_att(i, index.iz, index.iy, index.ix), mask, tag_type());
    }
  }

  /**
   * @brief Store evolved SLS memory variables from a point-local struct back
   *        to the device views.
   *
   * Only the memory variables and du field are written; relaxation rates are
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
      epsilon_xx_att(i, index.iz, index.iy, index.ix) = point.epsilon_xx;
      epsilon_yy_att(i, index.iz, index.iy, index.ix) = point.epsilon_yy;
      epsilon_zz_att(i, index.iz, index.iy, index.ix) = point.epsilon_zz;
      epsilon_xy_att(i, index.iz, index.iy, index.ix) = point.epsilon_xy;
      epsilon_xz_att(i, index.iz, index.iy, index.ix) = point.epsilon_xz;
      epsilon_yz_att(i, index.iz, index.iy, index.ix) = point.epsilon_yz;
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      const auto mask = index.template get_mask<simd>();
      for (int j = 0; j < N_SLS; ++j) {
        Kokkos::Experimental::simd_partial_store(
            point.Rxx(j),
            &memory_variable_Rxx(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Ryy(j),
            &memory_variable_Ryy(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Rxy(j),
            &memory_variable_Rxy(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Rxz(j),
            &memory_variable_Rxz(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Ryz(j),
            &memory_variable_Ryz(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Rkappa(j),
            &memory_variable_kappa(i, index.iz, index.iy, index.ix, j), mask,
            tag_type());
      }
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_xx, &epsilon_xx_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_yy, &epsilon_yy_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_zz, &epsilon_zz_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_xy, &epsilon_xy_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_xz, &epsilon_xz_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_yz, &epsilon_yz_att(i, index.iz, index.iy, index.ix),
          mask, tag_type());
    }
  }
};

} // namespace specfem::assembly::impl
