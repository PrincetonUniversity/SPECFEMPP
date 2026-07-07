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
#include <cstddef>
#include <string>

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

  // Host-only per-GLL modulus scale factors (runtime = physical * scale):
  // shape [nspec_attn][ngllz][ngllx]. Element-constant at construction but
  // stored per GLL so a GLL-varying Q model read from disk is honoured.
  scalar_view_type::host_mirror_type h_kappa_scale;
  scalar_view_type::host_mirror_type h_mu_scale;

  // Host-only per-GLL quality factors for model I/O; never copied to device.
  // Per-GLL (not per-element) for forward-compat with GLL-varying Q and
  // format uniformity with the property datasets.
  scalar_view_type::host_mirror_type h_Qkappa;
  scalar_view_type::host_mirror_type h_Qmu;

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
    h_kappa_scale = scalar_view_type::host_mirror_type(
        "h_kappa_scale", nspec_attn, ngllz, ngllx);
    h_mu_scale = scalar_view_type::host_mirror_type("h_mu_scale", nspec_attn,
                                                    ngllz, ngllx);
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

    h_Qkappa = scalar_view_type::host_mirror_type("h_Qkappa", nspec_attn, ngllz,
                                                  ngllx);
    h_Qmu = scalar_view_type::host_mirror_type("h_Qmu", nspec_attn, ngllz,
                                               ngllx);

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

      // Modulus scale factors and quality factors for model I/O. Q is
      // element-constant at construction but stored per GLL point (see
      // h_Qkappa above).
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int ix = 0; ix < ngllx; ++ix) {
          h_Qkappa(i, iz, ix) = material.Qkappa;
          h_Qmu(i, iz, ix) = material.Qmu;
          h_kappa_scale(i, iz, ix) = computed_values.kappa_scale;
          h_mu_scale(i, iz, ix) = computed_values.mu_scale;
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

  /**
   * @brief Recompute the per-GLL modulus scale factors from the current
   *        h_Qkappa/h_Qmu.
   *
   * Called by the property reader after Q has been read from disk so kappa/mu
   * can be scaled to their unrelaxed values with a factor derived from the
   * on-disk Q. Q is sampled at every GLL point, so a GLL-varying Q model is
   * honoured; the (expensive, Nelder-Mead) tau_epsilon solve is memoized and
   * only redone when Q changes from the previous point, so element-constant Q
   * costs one solve per element. Host-only; const because only view data is
   * mutated.
   *
   * @param fc Band-center frequency for modulus scaling
   * @param f0 Reference frequency
   * @param band Attenuation frequency band
   * @param tau_sigma Stress relaxation times
   */
  void recompute_scale_factors(
      const specfem::units::Hertz fc, const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) const {
    const int nspec_attn = element_range.size();
    const int l_ngllz = h_kappa_scale.extent(1);
    const int l_ngllx = h_kappa_scale.extent(2);
    for (int i = 0; i < nspec_attn; ++i) {
      type_real last_Qkappa = -1, last_Qmu = -1, kappa_scale = 0, mu_scale = 0;
      for (int iz = 0; iz < l_ngllz; ++iz) {
        for (int ix = 0; ix < l_ngllx; ++ix) {
          const type_real Qkappa = h_Qkappa(i, iz, ix);
          const type_real Qmu = h_Qmu(i, iz, ix);
          if (Qkappa != last_Qkappa) {
            kappa_scale =
                specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                    fc.raw(),
                    specfem::attenuation::compute_tau_eps<N_SLS>(
                        Qkappa, tau_sigma, band),
                    tau_sigma, Qkappa, f0.raw());
            last_Qkappa = Qkappa;
          }
          if (Qmu != last_Qmu) {
            mu_scale =
                specfem::attenuation::get_attenuation_scale_factor<N_SLS>(
                    fc.raw(),
                    specfem::attenuation::compute_tau_eps<N_SLS>(Qmu, tau_sigma,
                                                                 band),
                    tau_sigma, Qmu, f0.raw());
            last_Qmu = Qmu;
          }
          h_kappa_scale(i, iz, ix) = kappa_scale;
          h_mu_scale(i, iz, ix) = mu_scale;
        }
      }
    }
  }

  /**
   * @brief Recompute the per-GLL relaxation rates from the current
   *        h_Qkappa/h_Qmu and the supplied (unrelaxed) moduli, then push them
   *        to device.
   *
   * Mirrors the construction-time fill (`modulus * beta / (tau_sigma *
   * one_minus_sum_beta)`, mu doubled) but takes the unrelaxed moduli from the
   * property container after a model read, so a model that differs from the
   * mesh database produces consistent attenuation physics. Q is sampled at
   * every GLL point (GLL-varying Q models are honoured); the tau_epsilon
   * solve is memoized and only redone when Q changes from the previous point.
   *
   * @tparam KappaViewType Host kappa-modulus view type (group-local indexing)
   * @tparam MuViewType Host mu-modulus view type (group-local indexing)
   * @tparam ElementIndicesType Group-local index -> global ispec mapping type
   * @param h_kappa Unrelaxed kappa modulus host view (group-local)
   * @param h_mu Unrelaxed mu modulus host view (group-local)
   * @param element_indices Group-local element index -> global ispec mapping
   * @param band Attenuation frequency band
   * @param tau_sigma Stress relaxation times
   */
  template <typename KappaViewType, typename MuViewType,
            typename ElementIndicesType>
  void recompute_relaxation_rates(
      const KappaViewType &h_kappa, const MuViewType &h_mu,
      const ElementIndicesType &element_indices,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) const {
    const int nspec_attn = element_range.size();
    if (nspec_attn == 0)
      return;
    const int l_ngllz = h_kappa_relaxation_rate.extent(1);
    const int l_ngllx = h_kappa_relaxation_rate.extent(2);
    for (int gi = 0; gi < static_cast<int>(element_indices.size()); ++gi) {
      const int i = element_indices(gi) - element_range.begin_index();
      if (i < 0 || i >= nspec_attn)
        continue;
      type_real last_Qkappa = -1, last_Qmu = -1;
      specfem::attenuation::AttenuationPropertyValues<N_SLS> kappa_props,
          mu_props;
      for (int iz = 0; iz < l_ngllz; ++iz) {
        for (int ix = 0; ix < l_ngllx; ++ix) {
          const type_real Qkappa = h_Qkappa(i, iz, ix);
          const type_real Qmu = h_Qmu(i, iz, ix);
          if (Qkappa != last_Qkappa) {
            kappa_props =
                specfem::attenuation::get_attenuation_property_values<N_SLS>(
                    tau_sigma, specfem::attenuation::compute_tau_eps<N_SLS>(
                                   Qkappa, tau_sigma, band));
            last_Qkappa = Qkappa;
          }
          if (Qmu != last_Qmu) {
            mu_props =
                specfem::attenuation::get_attenuation_property_values<N_SLS>(
                    tau_sigma, specfem::attenuation::compute_tau_eps<N_SLS>(
                                   Qmu, tau_sigma, band));
            last_Qmu = Qmu;
          }
          for (int j = 0; j < N_SLS; ++j) {
            const type_real tauinv_j = 1.0 / tau_sigma(j);
            h_kappa_relaxation_rate(i, iz, ix, j) =
                h_kappa(gi, iz, ix) * kappa_props.beta(j) * tauinv_j /
                kappa_props.one_minus_sum_beta;
            h_mu_relaxation_rate(i, iz, ix, j) = h_mu(gi, iz, ix) * 2.0 *
                                                 mu_props.beta(j) * tauinv_j /
                                                 mu_props.one_minus_sum_beta;
          }
        }
      }
    }
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
  }

  // ---- Model-I/O interface (used by the property writer/reader) ----
  // All attenuation-type-specific knowledge (which property views are scaled,
  // which model datasets are persisted) lives here; the property writer/reader
  // stay agnostic of the attenuation implementation.

  /**
   * @brief Visit each attenuation model dataset as (host_view, name).
   *
   * Mirrors the property container's for_each_host_view; the property
   * writer/reader persist these datasets alongside the property views.
   * constant_isotropic persists the per-GLL quality factors Qkappa/Qmu.
   * No-op when there are no attenuating elements.
   *
   * @tparam Fn Callback type invoked as fn(host_view, name)
   * @param fn Callback invoked for each model dataset
   */
  template <typename Fn> void for_each_io_host_view(Fn &&fn) const {
    if (element_range.size() == 0)
      return;
    fn(h_Qkappa, std::string("Qkappa"));
    fn(h_Qmu, std::string("Qmu"));
  }

  /**
   * @brief Visit each host view the reader recomputes from the on-disk model
   *        as (host_view, name).
   *
   * State derived from the model datasets rather than persisted itself; all
   * views are element-major domain views. constant_isotropic yields the
   * per-GLL modulus scale factors and relaxation rates. No-op when there are
   * no attenuating elements.
   *
   * @tparam Fn Callback type invoked as fn(host_view, name)
   * @param fn Callback invoked for each recomputed view
   */
  template <typename Fn> void for_each_recomputed_host_view(Fn &&fn) const {
    if (element_range.size() == 0)
      return;
    fn(h_kappa_scale, std::string("kappa_scale"));
    fn(h_mu_scale, std::string("mu_scale"));
    fn(h_kappa_relaxation_rate, std::string("kappa_relaxation_rate"));
    fn(h_mu_relaxation_rate, std::string("mu_relaxation_rate"));
  }

  /**
   * @brief Return the view to persist for the named property view.
   *
   * The runtime property buffer stores the unrelaxed moduli (kappa * scale,
   * mu * scale) while the model file stores the PHYSICAL (relaxed) values.
   * For the views this attenuation type scales (kappa/mu), returns a scratch
   * copy holding view / scale (scale 1 for elements outside the attenuation
   * range); any other view -- and every view when there are no attenuating
   * elements -- is returned unchanged. The live buffer is never mutated (on
   * CPU builds the host mirror aliases the device view).
   *
   * @tparam ViewType Host property view type (group-local indexing)
   * @tparam ElementIndicesType Group-local index -> global ispec mapping type
   * @param view Runtime property host view
   * @param name Dataset name of the view (e.g. "kappa")
   * @param element_indices Group-local element index -> global ispec mapping
   * @return View holding the values to persist
   */
  template <typename ViewType, typename ElementIndicesType>
  ViewType physical_view(const ViewType &view, const std::string &name,
                         const ElementIndicesType &element_indices) const {
    if (element_range.size() == 0 || (name != "kappa" && name != "mu"))
      return view;
    const auto &scale = (name == "kappa") ? h_kappa_scale : h_mu_scale;
    ViewType scratch("physical_" + name, view.get_mapping());
    for (int gi = 0; gi < static_cast<int>(element_indices.size()); ++gi) {
      const int a = element_indices(gi) - element_range.begin_index();
      const bool attenuating = (a >= 0 && a < element_range.size());
      for (std::size_t iz = 0; iz < view.extent(1); ++iz)
        for (std::size_t ix = 0; ix < view.extent(2); ++ix) {
          const type_real s =
              attenuating ? scale(a, iz, ix) : static_cast<type_real>(1);
          scratch(gi, iz, ix) = view(gi, iz, ix) / s;
        }
    }
    return scratch;
  }

  /**
   * @brief Recompute runtime attenuation state after a model read.
   *
   * The model file stores physical (relaxed) moduli plus per-GLL Q; the
   * runtime needs unrelaxed moduli, per-GLL scale factors and per-GLL
   * relaxation rates. Recomputes the scale factors from the read-back Q,
   * scales kappa/mu in place inside @p props (physical -> unrelaxed), and
   * recomputes the relaxation rates from the unrelaxed moduli (pushed to
   * device). No-op when there are no attenuating elements.
   *
   * @tparam PropsContainer Property data container type (exposes h_kappa/h_mu)
   * @tparam ElementIndicesType Group-local index -> global ispec mapping type
   * @param props The just-read property container
   * @param element_indices Group-local element index -> global ispec mapping
   * @param fc Band-center frequency for modulus scaling
   * @param f0 Reference frequency
   * @param band Attenuation frequency band
   * @param tau_sigma Stress relaxation times
   */
  template <typename PropsContainer, typename ElementIndicesType>
  void recompute(
      const PropsContainer &props, const ElementIndicesType &element_indices,
      const specfem::units::Hertz fc, const specfem::units::Hertz f0,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) const {
    if (element_range.size() == 0)
      return;
    recompute_scale_factors(fc, f0, band, tau_sigma);
    scale_to_runtime(props.h_kappa, h_kappa_scale, element_indices);
    scale_to_runtime(props.h_mu, h_mu_scale, element_indices);
    recompute_relaxation_rates(props.h_kappa, props.h_mu, element_indices,
                               band, tau_sigma);
  }

  /**
   * @brief Scale a physical (relaxed) modulus view to runtime (unrelaxed)
   *        values in place: view *= scale for attenuating elements.
   *
   * @tparam ViewType Host modulus view type (group-local indexing)
   * @tparam ScaleViewType Per-GLL scale-factor view type
   * @tparam ElementIndicesType Group-local index -> global ispec mapping type
   * @param view Physical modulus host view, scaled in place
   * @param scale Per-GLL modulus scale factors (attenuation-local index)
   * @param element_indices Group-local element index -> global ispec mapping
   */
  template <typename ViewType, typename ScaleViewType,
            typename ElementIndicesType>
  void scale_to_runtime(const ViewType &view, const ScaleViewType &scale,
                        const ElementIndicesType &element_indices) const {
    for (int gi = 0; gi < static_cast<int>(element_indices.size()); ++gi) {
      const int a = element_indices(gi) - element_range.begin_index();
      if (a < 0 || a >= element_range.size())
        continue;
      for (std::size_t iz = 0; iz < view.extent(1); ++iz)
        for (std::size_t ix = 0; ix < view.extent(2); ++ix)
          view(gi, iz, ix) *= scale(a, iz, ix);
    }
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
