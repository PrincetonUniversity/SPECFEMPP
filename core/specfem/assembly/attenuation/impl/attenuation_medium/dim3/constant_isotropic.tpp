#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/datatype.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh/dim3/materials/materials.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <string>

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

  // Host-only per-GLL modulus scale factors (unrelaxed = physical * scale):
  // shape [nspec_attn][ngllz][nglly][ngllx]. Element-constant today (broadcast
  // to every GLL point) but stored per-GLL for forward-compat with GLL-varying
  // Q. Recomputed from Q + frequency band; used by model I/O to (un)scale
  // kappa/mu.
  scalar_view_type::host_mirror_type h_kappa_scale;
  scalar_view_type::host_mirror_type h_mu_scale;

  // Stashed frequency-band inputs so the scale factors can be recomputed from
  // (possibly edited) Q at model-read time (see recompute_scaling()).
  specfem::units::Hertz f0_{};
  specfem::units::Hertz fc_{};
  specfem::utilities::Band<specfem::units::Hertz> band_{};
  Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace> tau_sigma_;

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

  // Host-only per-GLL quality factors: shape [nspec_attn][ngllz][nglly][ngllx].
  // Stored for model I/O only -- Q is not used at runtime, so these are never
  // copied to device or loaded into the point-local attenuation struct.
  scalar_view_type::host_mirror_type h_Qkappa;
  scalar_view_type::host_mirror_type h_Qmu;

  specfem::datatype::ElementIndexRange element_range; ///< Global element index range for this type

  attenuation_medium() = default;

  attenuation_medium(
      const specfem::datatype::ElementIndexRange &elements,
      const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
          &mesh,
      const specfem::mesh::materials<specfem::element::dimension_tag::dim3>
          &materials,
      const int ngllz, const int nglly, const int ngllx,
      const specfem::units::Hertz fc, const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma,
      const bool has_gll_model) {

    const int nspec_attn = elements.extent(0);

    // Stash band inputs so scale factors can be recomputed from (edited) Q on
    // model read.
    f0_ = f0;
    fc_ = fc;
    band_ = band;
    tau_sigma_ = tau_sigma;

    // 1. Allocate all views
    h_kappa_scale = scalar_view_type::host_mirror_type(
        "h_kappa_scale", nspec_attn, ngllz, nglly, ngllx);
    h_mu_scale = scalar_view_type::host_mirror_type("h_mu_scale", nspec_attn,
                                                    ngllz, nglly, ngllx);
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

    // Host-only quality-factor views for model I/O (no device counterpart).
    h_Qkappa = scalar_view_type::host_mirror_type("h_Qkappa", nspec_attn, ngllz,
                                                  nglly, ngllx);
    h_Qmu = scalar_view_type::host_mirror_type("h_Qmu", nspec_attn, ngllz,
                                               nglly, ngllx);

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

    // When reading GLL model from disk, recompute() refills the per-GLL Q and
    // from Q computed attenuation values.
    if (has_gll_model) {
      return;
    }

    // 3. Loop over elements
    for (int i = 0; i < nspec_attn; ++i) {
      const int ispec = elements(i);
      const int mesh_ispec = mesh.h_compute_to_mesh(ispec);

      auto material = materials.template get_material<
          specfem::element::medium_tag::elastic, PropertyTag,
          specfem::element::attenuation_tag::constant_isotropic>(mesh_ispec);

      // Store the (element-constant) quality factors at every GLL point for
      // model I/O. Not used at runtime.
      const type_real Qkappa = material.Qkappa;
      const type_real Qmu = material.Qmu;
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            h_Qkappa(i, iz, iy, ix) = Qkappa;
            h_Qmu(i, iz, iy, ix) = Qmu;
          }
        }
      }

      auto computed_values = material.compute_attenuation_properties(
          f0.raw(), fc.raw(), band, tau_sigma);
      auto kappa_props = computed_values.kappa_attenuation_properties;
      auto mu_props = computed_values.mu_attenuation_properties;

      // Scaled moduli: matches SPECFEM3D's kappastore/mustore after
      // prepare_attenuation.f90 applies scale_factor in-place.
      const auto scaled_props = material.get_properties();
      const type_real kappa_sc = scaled_props.kappa();
      const type_real mu_sc = scaled_props.mu();

      // Store the (element-constant) modulus scale factors at every GLL point
      // for model I/O (lets the writer/reader (un)scale kappa/mu between the
      // physical (relaxed) and runtime (unrelaxed) representations).
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            h_kappa_scale(i, iz, iy, ix) = computed_values.kappa_scale;
            h_mu_scale(i, iz, iy, ix) = computed_values.mu_scale;
          }
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

  /**
   * @brief Recompute the per-GLL modulus scale factors from the current
   *        h_Qkappa/h_Qmu and the stored frequency band.
   *
   * Used by the property reader after Q has been read from disk, so kappa/mu are
   * scaled to their unrelaxed values using a factor derived from the on-disk Q.
   * Host-only; mirrors the scale computation in @ref ComputedAttenuationValues.
   * Q is sampled per GLL point, so a GLL-varying Q model is honoured. The
   * (expensive, Nelder-Mead) tau_epsilon solve is memoized and only redone when
   * Q changes from the previous point, so element-constant Q costs one solve
   * per element.
   */
  void recompute_scaling() {
    const int nspec_attn = h_kappa_scale.extent(0);
    const int l_ngllz = h_kappa_scale.extent(1);
    const int l_nglly = h_kappa_scale.extent(2);
    const int l_ngllx = h_kappa_scale.extent(3);
    for (int i = 0; i < nspec_attn; ++i) {
      type_real last_Qkappa = -1, last_Qmu = -1, kappa_scale = 0, mu_scale = 0;
      for (int iz = 0; iz < l_ngllz; ++iz) {
        for (int iy = 0; iy < l_nglly; ++iy) {
          for (int ix = 0; ix < l_ngllx; ++ix) {
            const type_real Qkappa = h_Qkappa(i, iz, iy, ix);
            const type_real Qmu = h_Qmu(i, iz, iy, ix);
            if (Qkappa != last_Qkappa) {
              kappa_scale =
                  specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                      fc_.raw(),
                      specfem::attenuation::compute_tau_eps<N_SLS>(
                          Qkappa, tau_sigma_, band_),
                      tau_sigma_, Qkappa, f0_.raw());
              last_Qkappa = Qkappa;
            }
            if (Qmu != last_Qmu) {
              mu_scale =
                  specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                      fc_.raw(),
                      specfem::attenuation::compute_tau_eps<N_SLS>(
                          Qmu, tau_sigma_, band_),
                      tau_sigma_, Qmu, f0_.raw());
              last_Qmu = Qmu;
            }
            h_kappa_scale(i, iz, iy, ix) = kappa_scale;
            h_mu_scale(i, iz, iy, ix) = mu_scale;
          }
        }
      }
    }
  }

  /**
   * @brief Recompute the per-GLL relaxation rates from the current h_Qkappa/
   *        h_Qmu and the supplied (unrelaxed) moduli, after a model read.
   *
   * The relaxation rate is `modulus * beta / (tau_sigma * one_minus_sum_beta)`
   * with `beta` derived from the on-disk Q -- mirrors the construction-time
   * fill, so a write/read round-trip reproduces the original rates. The
   * unrelaxed moduli come from the property container (after rescale_read). The
   * recomputed host views are pushed to device for use at runtime.
   *
   * @tparam KappaView    Host kappa-modulus view type (group-local indexing).
   * @tparam MuView       Host mu-modulus view type (group-local indexing).
   * @tparam ElementsView Group-local index -> global ispec view type.
   * @param h_kappa  Unrelaxed kappa modulus host view (group-local).
   * @param h_mu     Unrelaxed mu modulus host view (group-local).
   * @param elements Group-local element index -> global ispec mapping.
   */
  template <typename KappaView, typename MuView, typename ElementsView>
  void recompute_relaxation_rates(const KappaView &h_kappa, const MuView &h_mu,
                                  const ElementsView &elements) {
    if (h_Qkappa.extent(0) == 0)
      return;
    for (std::size_t gi = 0; gi < h_kappa.extent(0); ++gi) {
      const int a = elements(gi) - element_range.begin_index();
      if (a < 0 || a >= static_cast<int>(h_Qkappa.extent(0)))
        continue;
      // Q is sampled per GLL point so a GLL-varying Q model is honoured. The
      // (expensive, Nelder-Mead) tau_epsilon solve is memoized and only redone
      // when Q changes, so element-constant Q costs one solve per element.
      type_real last_Qkappa = -1, last_Qmu = -1;
      specfem::attenuation::AttenuationPropertyValues<N_SLS> kappa_props,
          mu_props;
      for (std::size_t iz = 0; iz < h_kappa.extent(1); ++iz)
        for (std::size_t iy = 0; iy < h_kappa.extent(2); ++iy)
          for (std::size_t ix = 0; ix < h_kappa.extent(3); ++ix) {
            const type_real Qkappa = h_Qkappa(a, iz, iy, ix);
            const type_real Qmu = h_Qmu(a, iz, iy, ix);
            if (Qkappa != last_Qkappa) {
              kappa_props =
                  specfem::attenuation::get_attenuation_property_values<N_SLS>(
                      tau_sigma_, specfem::attenuation::compute_tau_eps<N_SLS>(
                                      Qkappa, tau_sigma_, band_));
              last_Qkappa = Qkappa;
            }
            if (Qmu != last_Qmu) {
              mu_props =
                  specfem::attenuation::get_attenuation_property_values<N_SLS>(
                      tau_sigma_, specfem::attenuation::compute_tau_eps<N_SLS>(
                                      Qmu, tau_sigma_, band_));
              last_Qmu = Qmu;
            }
            const type_real kappa_sc = h_kappa(gi, iz, iy, ix);
            const type_real mu_sc = h_mu(gi, iz, iy, ix);
            for (int j = 0; j < N_SLS; ++j) {
              const type_real tauinv_j = 1.0 / tau_sigma_(j);
              h_kappa_relaxation_rate(a, iz, iy, ix, j) =
                  kappa_sc * kappa_props.beta(j) * tauinv_j /
                  kappa_props.one_minus_sum_beta;
              h_mu_relaxation_rate(a, iz, iy, ix, j) =
                  mu_sc * 2.0 * mu_props.beta(j) * tauinv_j /
                  mu_props.one_minus_sum_beta;
            }
          }
    }
    specfem::datatype::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    specfem::datatype::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
  }

  /**
   * @brief Recompute all read-time attenuation state from the on-disk Q model.
   *
   * Single, implementation-agnostic entry point for the property reader: it
   * recomputes the modulus scale factors from the read-back Q, un-relaxes the
   * scaled property moduli in place, and recomputes the relaxation rates. A
   * no-op when there are no attenuating elements.
   *
   * @tparam PropsContainer The property data container (exposes
   *                        for_each_host_view and h_kappa/h_mu).
   * @tparam ElementsView   Group-local index -> global ispec view type.
   * @param props    The just-read property container; its scaled moduli are
   *                 un-relaxed in place.
   * @param elements Group-local element index -> global ispec mapping.
   */
  template <typename PropsContainer, typename ElementsView>
  void recompute(const PropsContainer &props, const ElementsView &elements) {
    if (!has_attenuating_elements())
      return;
    // 1. Scale factors from the read-back Q.
    recompute_scaling();
    // 2. Un-relax the scaled moduli (physical -> unrelaxed) in place.
    props.for_each_host_view([&](const auto &view, const std::string &name) {
      if (is_scaled_property(name))
        scale_into(view, view, name, /*to_physical=*/false, elements);
    });
    // 3. Relaxation rates from the unrelaxed moduli + Q.
    recompute_relaxation_rates(props.h_kappa, props.h_mu, elements);
  }

  // ---- Model-I/O interface (consumed by specfem::io::impl::AttenuationIO) ----

  /**
   * @brief Whether this container has any attenuating elements to persist.
   *
   * @return True if at least one attenuating element is stored.
   */
  bool has_attenuating_elements() const { return h_Qkappa.extent(0) != 0; }

  /**
   * @brief Whether the named property view carries a per-GLL modulus scale
   *        (i.e. must be (un)scaled between physical and runtime values).
   *
   * @param name Property view name.
   * @return True for the scaled moduli ("kappa", "mu").
   */
  bool is_scaled_property(const std::string &name) const {
    return name == "kappa" || name == "mu";
  }

  /**
   * @brief Visit each persisted model-I/O dataset as (host_view, name).
   *
   * Mirrors the property container's for_each_host_view.
   *
   * @tparam Fn Callable invoked as fn(host_view, name).
   * @param fn Visitor applied to each (view, name) pair.
   */
  template <typename Fn> void for_each_io_host_view(Fn &&fn) const {
    fn(h_Qkappa, std::string("Qkappa"));
    fn(h_Qmu, std::string("Qmu"));
  }

  /**
   * @brief Convert a named modulus view between the physical (relaxed) and
   *        runtime (unrelaxed) representations: dst = to_physical ? src/scale
   *        : src*scale.
   *
   * The loop runs over the (medium, property) group, a superset of the
   * attenuating elements; rows outside the attenuation sub-range (compact index
   * out of [0, nspec_attn)) are copied unchanged. @p dst may alias @p src
   * (in-place read re-scale).
   *
   * @tparam DstView      Destination host view type.
   * @tparam SrcView      Source host view type.
   * @tparam ElementsView Group-local index -> global ispec view type.
   * @param dst         Destination view (may alias @p src).
   * @param src         Source modulus view.
   * @param name        Property name ("kappa" or "mu").
   * @param to_physical Divide by scale when true, multiply when false.
   * @param elements    Group-local index -> global ispec (from element_types).
   */
  template <typename DstView, typename SrcView, typename ElementsView>
  void scale_into(const DstView &dst, const SrcView &src,
                  const std::string &name, const bool to_physical,
                  const ElementsView &elements) const {
    const auto &scale = (name == "kappa") ? h_kappa_scale : h_mu_scale;
    for (std::size_t i = 0; i < src.extent(0); ++i) {
      const int a = elements(i) - element_range.begin_index();
      const bool attenuating = (a >= 0 && a < static_cast<int>(scale.extent(0)));
      for (std::size_t iz = 0; iz < src.extent(1); ++iz)
        for (std::size_t iy = 0; iy < src.extent(2); ++iy)
          for (std::size_t ix = 0; ix < src.extent(3); ++ix) {
            const type_real s =
                attenuating ? scale(a, iz, iy, ix) : static_cast<type_real>(1);
            const type_real v = src(i, iz, iy, ix);
            dst(i, iz, iy, ix) = to_physical ? v / s : v * s;
          }
    }
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
   *
   * @tparam IndexType Point index type (provides ispec/iz/iy/ix and SIMD mask).
   * @tparam PointType Point-local attenuation struct type.
   * @param index Point index identifying the GLL location.
   * @param point Output struct populated with relaxation rates and memory vars.
   */
  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void load_device_values(const IndexType &index,
                                                 PointType &point) const {
    const int i = index.ispec - element_range.begin_index();
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
   *
   * @tparam IndexType Point index type (provides ispec/iz/iy/ix and SIMD mask).
   * @tparam PointType Point-local attenuation struct type.
   * @param index Point index identifying the GLL location.
   * @param point Source struct holding the evolved memory variables.
   */
  template <typename IndexType, typename PointType>
  KOKKOS_INLINE_FUNCTION void
  store_device_values(const IndexType &index, const PointType &point) const {
    const int i = index.ispec - element_range.begin_index();
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
