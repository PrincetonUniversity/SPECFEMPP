#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh/dim2/materials/materials.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/logarithmic_center.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace specfem::assembly::impl {

template <specfem::element::property_tag PropertyTag>
struct attenuation_medium<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          PropertyTag,
                          specfem::element::attenuation_tag::constant_isotropic>
    : specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::attenuation,
          specfem::element::dimension_tag::dim2> {

  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim2>;

  using view_type = typename base_type::vector_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  constexpr static int N_SLS = specfem::constants::N_SLS;

  // Scalar-per-GLL view: shape [nspec_attn][ngllz][ngllx]
  using scalar_view_type = typename base_type::template scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  // Host-only per-element scale factors: unrelaxed = physical * scale
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_kappa_scale;
  Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> h_mu_scale;

  // Host-only per-GLL quality factors, persisted by the property
  // writer/reader. Per-GLL (not per-element) for forward-compat with
  // GLL-varying Q and format uniformity with the property datasets. Plain
  // layout (not tiled) so they serialize directly; never copied to device.
  using q_view_type =
      Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>;
  q_view_type h_Qkappa;
  q_view_type h_Qmu;

  // Views: shape [nspec_attn][ngllz][ngllx][N_SLS]
  view_type kappa_relaxation_rate;
  view_type::host_mirror_type h_kappa_relaxation_rate;
  view_type mu_relaxation_rate;
  view_type::host_mirror_type h_mu_relaxation_rate;
  view_type memory_variable_kappa;
  view_type::host_mirror_type h_memory_variable_kappa;
  view_type memory_variable_Rxx;
  view_type::host_mirror_type h_memory_variable_Rxx;
  view_type memory_variable_Rxz;
  view_type::host_mirror_type h_memory_variable_Rxz;

  // Symmetrised strain components from previous Taylor step: shape
  // [nspec_attn][ngllz][ngllx]
  scalar_view_type epsilon_xx_att;
  scalar_view_type::host_mirror_type h_epsilon_xx_att;
  scalar_view_type epsilon_zz_att;
  scalar_view_type::host_mirror_type h_epsilon_zz_att;
  scalar_view_type epsilon_xz_att;
  scalar_view_type::host_mirror_type h_epsilon_xz_att;

  specfem::datatype::ElementIndexRange element_range; ///< Global element index range for this type

  attenuation_medium() = default;

  attenuation_medium(
      const specfem::datatype::ElementIndexRange &elements,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim2>
          &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim2>
          &materials,
      const int ngllz, const int ngllx, const specfem::units::Hertz fc,
      const specfem::units::Hertz f0,
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
    h_Qkappa = q_view_type("h_Qkappa", nspec_attn, ngllz, ngllx);
    h_Qmu = q_view_type("h_Qmu", nspec_attn, ngllz, ngllx);
    kappa_relaxation_rate =
        view_type("kappa_relaxation_rate", nspec_attn, ngllz, ngllx, N_SLS);
    h_kappa_relaxation_rate =
        specfem::datatype::create_mirror_view(kappa_relaxation_rate);
    mu_relaxation_rate =
        view_type("mu_relaxation_rate", nspec_attn, ngllz, ngllx, N_SLS);
    h_mu_relaxation_rate =
        specfem::datatype::create_mirror_view(mu_relaxation_rate);
    memory_variable_kappa =
        view_type("mem_kappa", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_kappa =
        specfem::datatype::create_mirror_view(memory_variable_kappa);
    memory_variable_Rxx = view_type("mem_Rxx", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_Rxx =
        specfem::datatype::create_mirror_view(memory_variable_Rxx);
    memory_variable_Rxz = view_type("mem_Rxz", nspec_attn, ngllz, ngllx, N_SLS);
    h_memory_variable_Rxz =
        specfem::datatype::create_mirror_view(memory_variable_Rxz);

    epsilon_xx_att =
        scalar_view_type("epsilon_xx_att", nspec_attn, ngllz, ngllx);
    h_epsilon_xx_att = specfem::datatype::create_mirror_view(epsilon_xx_att);
    Kokkos::deep_copy(epsilon_xx_att, static_cast<type_real>(0));
    epsilon_zz_att =
        scalar_view_type("epsilon_zz_att", nspec_attn, ngllz, ngllx);
    h_epsilon_zz_att = specfem::datatype::create_mirror_view(epsilon_zz_att);
    Kokkos::deep_copy(epsilon_zz_att, static_cast<type_real>(0));
    epsilon_xz_att =
        scalar_view_type("epsilon_xz_att", nspec_attn, ngllz, ngllx);
    h_epsilon_xz_att = specfem::datatype::create_mirror_view(epsilon_xz_att);
    Kokkos::deep_copy(epsilon_xz_att, static_cast<type_real>(0));

    element_range = elements;

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
          specfem::element::medium_tag::elastic_psv, PropertyTag,
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

      h_kappa_scale(i) = computed_values.kappa_scale;
      h_mu_scale(i) = computed_values.mu_scale;

      // Quality factors: element-constant at construction but stored per GLL
      // point (see h_Qkappa above).
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          h_Qkappa(i, iz, ix) = material.Qkappa;
          h_Qmu(i, iz, ix) = material.Qmu;
        }
      }

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
          for (int ix = 0; ix < ngllx; ++ix) {
            h_kappa_relaxation_rate(i, iz, ix, j) = kappa_rr_j;
            h_mu_relaxation_rate(i, iz, ix, j) = mu_rr_j;
          }
        }
      }
    }

    // 4. Push all host data (kappa/mu_cf filled; memory variables zero) to
    // device
    copy_to_device();
  }

  // ---- Model-I/O interface (used by the property writer/reader) ----

  /**
   * @brief Transform staged property values from unrelaxed (runtime) to
   *        physical (relaxed) in place.
   *
   * The property container holds unrelaxed moduli for attenuating elements;
   * the property file stores the physical model. Divides "kappa"/"mu" by the
   * per-element scale factors; any other dataset name is left untouched.
   *
   * Precondition: the element axis of @p scratch aligns 1:1 with this
   * container's element_range (one file group per (medium, property,
   * attenuation) combination guarantees this).
   *
   * @tparam ViewType Plain host view of shape [nspec_attn][ngllz][ngllx]
   * @param scratch Staged copy of a property dataset, modified in place
   * @param name Property dataset name (as visited by for_each_host_view)
   */
  template <typename ViewType>
  void to_physical(const ViewType &scratch, const std::string &name) const {
    if (name != "kappa" && name != "mu") {
      return;
    }
    const auto &scale = (name == "kappa") ? h_kappa_scale : h_mu_scale;
    for (int i = 0; i < static_cast<int>(scratch.extent(0)); ++i) {
      for (int iz = 0; iz < static_cast<int>(scratch.extent(1)); ++iz) {
        for (int ix = 0; ix < static_cast<int>(scratch.extent(2)); ++ix) {
          scratch(i, iz, ix) /= scale(i);
        }
      }
    }
  }

  /**
   * @brief The attenuation model datasets as (name, host_view) pairs.
   *
   * The property writer/reader persist these verbatim alongside the property
   * datasets of the same file group. Call only when element_range is
   * non-empty.
   *
   * @return Array of (dataset name, host view) pairs
   */
  std::array<std::pair<std::string, q_view_type>, 2> get_views() const {
    return { { { "Qkappa", h_Qkappa }, { "Qmu", h_Qmu } } };
  }

  /**
   * @brief Recompute runtime attenuation state after a model read.
   *
   * The property reader leaves the physical (relaxed) moduli from the file in
   * @p props and the per-GLL Q in h_Qkappa/h_Qmu. The runtime needs unrelaxed
   * moduli and relaxation rates: for every attenuating GLL point a modulus
   * scale factor is derived from the read-back Q, the moduli in @p props are
   * scaled to unrelaxed in place, and the relaxation rates are recomputed
   * from them (pushed to device). The per-element scale factors are refreshed
   * so a subsequent write stays consistent with the read model. Q is sampled
   * at every GLL point, so a GLL-varying Q model is honoured; the (expensive,
   * Nelder-Mead) tau_epsilon solve is memoized and only redone when Q changes
   * from the previous point, so element-constant Q costs one solve per
   * element. No-op when there are no attenuating elements. Host-only; const
   * because only view data is mutated.
   *
   * @tparam PropsContainer Property data container type (exposes h_kappa,
   *                        h_mu and element_range)
   * @param props The just-read property container
   * @param fc Band-center frequency for modulus scaling
   * @param f0 Reference frequency
   * @param band Attenuation frequency band
   * @param tau_sigma Stress relaxation times
   */
  template <typename PropsContainer>
  void recompute(
      const PropsContainer &props, const specfem::units::Hertz fc,
      const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) const {
    if (element_range.size() == 0) {
      return;
    }
    const int nspec_attn = element_range.size();
    const int offset =
        element_range.begin_index() - props.element_range.begin_index();
    const int l_ngllz = h_Qkappa.extent(1);
    const int l_ngllx = h_Qkappa.extent(2);
    for (int i = 0; i < nspec_attn; ++i) {
      type_real last_Qkappa = -1, last_Qmu = -1, kappa_scale = 0, mu_scale = 0;
      specfem::attenuation::AttenuationPropertyValues<N_SLS> kappa_props,
          mu_props;
      for (int iz = 0; iz < l_ngllz; ++iz) {
        for (int ix = 0; ix < l_ngllx; ++ix) {
          const type_real Qkappa = h_Qkappa(i, iz, ix);
          const type_real Qmu = h_Qmu(i, iz, ix);
          if (Qkappa != last_Qkappa) {
            const auto tau_eps = specfem::attenuation::compute_tau_eps<N_SLS>(
                Qkappa, tau_sigma, band);
            kappa_scale =
                specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                    fc.raw(), tau_eps, tau_sigma, Qkappa, f0.raw());
            kappa_props =
                specfem::attenuation::get_attenuation_property_values<N_SLS>(
                    tau_sigma, tau_eps);
            last_Qkappa = Qkappa;
          }
          if (Qmu != last_Qmu) {
            const auto tau_eps = specfem::attenuation::compute_tau_eps<N_SLS>(
                Qmu, tau_sigma, band);
            mu_scale =
                specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                    fc.raw(), tau_eps, tau_sigma, Qmu, f0.raw());
            mu_props =
                specfem::attenuation::get_attenuation_property_values<N_SLS>(
                    tau_sigma, tau_eps);
            last_Qmu = Qmu;
          }
          const type_real kappa_unrelaxed =
              props.h_kappa(offset + i, iz, ix) * kappa_scale;
          const type_real mu_unrelaxed =
              props.h_mu(offset + i, iz, ix) * mu_scale;
          props.h_kappa(offset + i, iz, ix) = kappa_unrelaxed;
          props.h_mu(offset + i, iz, ix) = mu_unrelaxed;
          for (int j = 0; j < N_SLS; ++j) {
            const type_real tauinv_j = 1.0 / tau_sigma(j);
            h_kappa_relaxation_rate(i, iz, ix, j) =
                kappa_unrelaxed * kappa_props.beta(j) * tauinv_j /
                kappa_props.one_minus_sum_beta;
            h_mu_relaxation_rate(i, iz, ix, j) = mu_unrelaxed * 2.0 *
                                                 mu_props.beta(j) * tauinv_j /
                                                 mu_props.one_minus_sum_beta;
          }
        }
      }
      // Refresh per-element scale factors (exact for element-constant Q; for
      // a GLL-varying model the last GLL point's value is kept).
      h_kappa_scale(i) = kappa_scale;
      h_mu_scale(i) = mu_scale;
    }
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
  }

  void copy_to_host() {
    Kokkos::deep_copy(h_kappa_relaxation_rate, kappa_relaxation_rate);
    Kokkos::deep_copy(h_mu_relaxation_rate, mu_relaxation_rate);
    Kokkos::deep_copy(h_memory_variable_kappa, memory_variable_kappa);
    Kokkos::deep_copy(h_memory_variable_Rxx, memory_variable_Rxx);
    Kokkos::deep_copy(h_memory_variable_Rxz, memory_variable_Rxz);
    Kokkos::deep_copy(h_epsilon_xx_att, epsilon_xx_att);
    Kokkos::deep_copy(h_epsilon_zz_att, epsilon_zz_att);
    Kokkos::deep_copy(h_epsilon_xz_att, epsilon_xz_att);
  }

  void copy_to_device() {
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
    Kokkos::deep_copy(memory_variable_kappa, h_memory_variable_kappa);
    Kokkos::deep_copy(memory_variable_Rxx, h_memory_variable_Rxx);
    Kokkos::deep_copy(memory_variable_Rxz, h_memory_variable_Rxz);
    Kokkos::deep_copy(epsilon_xx_att, h_epsilon_xx_att);
    Kokkos::deep_copy(epsilon_zz_att, h_epsilon_zz_att);
    Kokkos::deep_copy(epsilon_xz_att, h_epsilon_xz_att);
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
    const int i = index.ispec - element_range.begin_index();
    if constexpr (!IndexType::using_simd) {
      for (int j = 0; j < N_SLS; ++j) {
        point.kappa_relaxation_rate(j) =
            kappa_relaxation_rate(i, index.iz, index.ix, j);
        point.mu_relaxation_rate(j) =
            mu_relaxation_rate(i, index.iz, index.ix, j);
        point.Rxx(j) = memory_variable_Rxx(i, index.iz, index.ix, j);
        point.Rxz(j) = memory_variable_Rxz(i, index.iz, index.ix, j);
        point.Rkappa(j) = memory_variable_kappa(i, index.iz, index.ix, j);
      }
      point.epsilon_xx = epsilon_xx_att(i, index.iz, index.ix);
      point.epsilon_zz = epsilon_zz_att(i, index.iz, index.ix);
      point.epsilon_xz = epsilon_xz_att(i, index.iz, index.ix);
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      const auto mask = index.template get_mask<simd>();
      for (int j = 0; j < N_SLS; ++j) {
        point.kappa_relaxation_rate(j) =
            Kokkos::Experimental::simd_partial_load(
                &kappa_relaxation_rate(i, index.iz, index.ix, j), mask,
                tag_type());
        point.mu_relaxation_rate(j) = Kokkos::Experimental::simd_partial_load(
            &mu_relaxation_rate(i, index.iz, index.ix, j), mask, tag_type());
        point.Rxx(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Rxx(i, index.iz, index.ix, j), mask, tag_type());
        point.Rxz(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_Rxz(i, index.iz, index.ix, j), mask, tag_type());
        point.Rkappa(j) = Kokkos::Experimental::simd_partial_load(
            &memory_variable_kappa(i, index.iz, index.ix, j), mask, tag_type());
      }
      point.epsilon_xx = Kokkos::Experimental::simd_partial_load(
          &epsilon_xx_att(i, index.iz, index.ix), mask, tag_type());
      point.epsilon_zz = Kokkos::Experimental::simd_partial_load(
          &epsilon_zz_att(i, index.iz, index.ix), mask, tag_type());
      point.epsilon_xz = Kokkos::Experimental::simd_partial_load(
          &epsilon_xz_att(i, index.iz, index.ix), mask, tag_type());
    }
  }

  /**
   * @brief Store evolved SLS memory variables from a point-local struct back
   *        to the device views.
   *
   * Only the memory variables (Rxx, Rxz, Rkappa) and du field are written;
   * relaxation rates are simulation-lifetime constants and are not written
   * back.
   */
  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointType &point) const {
    const int i = index.ispec - element_range.begin_index();
    if constexpr (!IndexType::using_simd) {
      for (int j = 0; j < N_SLS; ++j) {
        memory_variable_Rxx(i, index.iz, index.ix, j) = point.Rxx(j);
        memory_variable_Rxz(i, index.iz, index.ix, j) = point.Rxz(j);
        memory_variable_kappa(i, index.iz, index.ix, j) = point.Rkappa(j);
      }
      epsilon_xx_att(i, index.iz, index.ix) = point.epsilon_xx;
      epsilon_zz_att(i, index.iz, index.ix) = point.epsilon_zz;
      epsilon_xz_att(i, index.iz, index.ix) = point.epsilon_xz;
    } else {
      using simd = typename PointType::simd;
      using mask_type = typename simd::mask_type;
      using tag_type = typename simd::tag_type;
      const auto mask = index.template get_mask<simd>();
      for (int j = 0; j < N_SLS; ++j) {
        Kokkos::Experimental::simd_partial_store(
            point.Rxx(j), &memory_variable_Rxx(i, index.iz, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Rxz(j), &memory_variable_Rxz(i, index.iz, index.ix, j), mask,
            tag_type());
        Kokkos::Experimental::simd_partial_store(
            point.Rkappa(j), &memory_variable_kappa(i, index.iz, index.ix, j),
            mask, tag_type());
      }
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_xx, &epsilon_xx_att(i, index.iz, index.ix), mask,
          tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_zz, &epsilon_zz_att(i, index.iz, index.ix), mask,
          tag_type());
      Kokkos::Experimental::simd_partial_store(
          point.epsilon_xz, &epsilon_xz_att(i, index.iz, index.ix), mask,
          tag_type());
    }
  }
};

} // namespace specfem::assembly::impl
