#pragma once

#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium/dim3/modulus_shift.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium/dim3/unrelaxed_tensor.hpp"
#include "specfem/attenuation/globe.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/attenuation/compute_factors.hpp"
#include "specfem/attenuation/compute_tau_eps.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/data_access/container.hpp"
#include "specfem/datatype.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/logger.hpp"
#include "specfem/mesh/dim3/materials/materials.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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

  // Host-only per-GLL quality factors, persisted by the property
  // writer/reader. Per-GLL (not per-element) for forward-compat with
  // GLL-varying Q and format uniformity with the property datasets. Plain
  // layout (not tiled) so they serialize directly; never copied to device.
  using q_view_type =
      Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  q_view_type h_Qkappa;
  q_view_type h_Qmu;

  // Host-only per-GLL modulus scale factors: unrelaxed = physical * scale.
  // Per-GLL so the property writer recovers the physical model exactly under
  // a GLL-varying Q.
  q_view_type h_kappa_scale;
  q_view_type h_mu_scale;

  // Attenuate kappa as well as mu. When false (globe meshes, as in globe)
  // kappa keeps scale 1 and zero relaxation rates, so the kappa memory
  // variables never leave zero.
  bool bulk_attenuation = true;

  // Host-only shift increments let model I/O recover the physical cij without
  // retaining the setup-only spherical coordinate cache.
  using correction_view_type =
      Kokkos::View<type_real *****, Kokkos::LayoutRight, Kokkos::HostSpace>;
  correction_view_type h_cij_correction;

  // Set once recompute() has run. A View so that copies of this container
  // share the flag.
  Kokkos::View<bool, Kokkos::HostSpace> h_recomputed;

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

  specfem::datatype::ElementIndexRange element_range; ///< Global element index
                                                      ///< range for this type

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
      const bool bulk_attenuation = true)
      : bulk_attenuation(bulk_attenuation) {

    const int nspec_attn = elements.extent(0);

    // 1. Allocate all views
    h_recomputed = Kokkos::View<bool, Kokkos::HostSpace>("recomputed");
    h_kappa_scale = q_view_type("kappa_scale", nspec_attn, ngllz, nglly, ngllx);
    h_mu_scale = q_view_type("mu_scale", nspec_attn, ngllz, nglly, ngllx);
    h_Qkappa = q_view_type("h_Qkappa", nspec_attn, ngllz, nglly, ngllx);
    h_Qmu = q_view_type("h_Qmu", nspec_attn, ngllz, nglly, ngllx);
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

    if constexpr (PropertyTag == specfem::element::property_tag::anisotropic) {
      h_cij_correction = correction_view_type("cij_correction", 21, nspec_attn,
                                              ngllz, nglly, ngllx);
      // The moduli are shifted by recompute() once the property container is
      // filled: by the globe oracle, the property reader, or (Cartesian
      // meshes) the assembly from these materials.
      for (int i = 0; i < nspec_attn; ++i) {
        const auto &material = materials.template get_material<
            specfem::element::medium_tag::elastic, PropertyTag,
            specfem::element::attenuation_tag::constant_isotropic>(
            mesh.h_compute_to_mesh(elements(i)));
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int iy = 0; iy < nglly; ++iy) {
            for (int ix = 0; ix < ngllx; ++ix) {
              h_Qkappa(i, iz, iy, ix) = material.Qkappa;
              h_Qmu(i, iz, iy, ix) = material.Qmu;
              h_kappa_scale(i, iz, iy, ix) = 1;
              h_mu_scale(i, iz, iy, ix) = 1;
            }
          }
        }
      }
    } else {
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
        // Without bulk attenuation kappa is not shifted and has no relaxation.
        // Only globe meshes disable it, and they overwrite the placeholder
        // material moduli from the model oracle before recompute().
        const type_real kappa_scale =
            bulk_attenuation ? computed_values.kappa_scale : 1.0;

        // Quality factors and scale factors: element-constant at construction
        // but stored per GLL point (see h_Qkappa above).
        for (int iz = 0; iz < ngllz; ++iz) {
          for (int iy = 0; iy < nglly; ++iy) {
            for (int ix = 0; ix < ngllx; ++ix) {
              h_Qkappa(i, iz, iy, ix) = material.Qkappa;
              h_Qmu(i, iz, iy, ix) = material.Qmu;
              h_kappa_scale(i, iz, iy, ix) = kappa_scale;
              h_mu_scale(i, iz, iy, ix) = computed_values.mu_scale;
            }
          }
        }

        // Per-GLL fill
        for (int j = 0; j < N_SLS; ++j) {

          // Compute per element relaxation rates (modulus included, matching
          // SPECFEM3D's factor_loc = modulus * factor_common)
          const type_real tauinv_j = 1.0 / tau_sigma(j);
          auto kappa_rr_j = bulk_attenuation
                                ? kappa_sc * kappa_props.beta(j) * tauinv_j /
                                      kappa_props.one_minus_sum_beta
                                : static_cast<type_real>(0);
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
   * per-GLL scale factors. For anisotropic properties subtracts the stored
   * Cartesian stiffness increments from each cij dataset. Other names are
   * left untouched.
   *
   * Precondition: the element axis of @p scratch aligns 1:1 with this
   * container's element_range (one file group per (medium, property,
   * attenuation) combination guarantees this).
   *
   * @tparam ViewType Plain host view of shape [nspec_attn][ngllz][nglly][ngllx]
   * @param scratch Staged copy of a property dataset, modified in place
   * @param name Property dataset name (as visited by for_each_host_view)
   */
  template <typename ViewType>
  void to_physical(const ViewType &scratch, const std::string &name) const {
    if constexpr (PropertyTag == specfem::element::property_tag::anisotropic) {
      constexpr std::array<const char *, 21> names = {
        "c11", "c12", "c13", "c14", "c15", "c16", "c22",
        "c23", "c24", "c25", "c26", "c33", "c34", "c35",
        "c36", "c44", "c45", "c46", "c55", "c56", "c66"
      };
      for (int k = 0; k < 21; ++k) {
        if (name != names[k])
          continue;
        for (int i = 0; i < static_cast<int>(scratch.extent(0)); ++i)
          for (int iz = 0; iz < static_cast<int>(scratch.extent(1)); ++iz)
            for (int iy = 0; iy < static_cast<int>(scratch.extent(2)); ++iy)
              for (int ix = 0; ix < static_cast<int>(scratch.extent(3)); ++ix)
                scratch(i, iz, iy, ix) -= h_cij_correction(k, i, iz, iy, ix);
      }
      return;
    }
    if (name != "kappa" && name != "mu") {
      return;
    }
    const auto &scale = (name == "kappa") ? h_kappa_scale : h_mu_scale;
    for (int i = 0; i < static_cast<int>(scratch.extent(0)); ++i) {
      for (int iz = 0; iz < static_cast<int>(scratch.extent(1)); ++iz) {
        for (int iy = 0; iy < static_cast<int>(scratch.extent(2)); ++iy) {
          for (int ix = 0; ix < static_cast<int>(scratch.extent(3)); ++ix) {
            scratch(i, iz, iy, ix) /= scale(i, iz, iy, ix);
          }
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
   * The property reader leaves the physical (reference-frequency) moduli
   * from the file in @p props and the per-GLL Q in h_Qkappa/h_Qmu. For every
   * attenuating GLL point this fits Q, shifts the moduli in @p props to
   * unrelaxed in place, refreshes the per-GLL scales (and, for anisotropic
   * properties, the cij increments) so a later write recovers the physical
   * model, and sets the relaxation rates from the unrelaxed moduli (pushed
   * to device). Q is sampled at every GLL point, so a GLL-varying Q model is
   * honoured; fits are memoized by a specfem::attenuation::QualityFactorTable,
   * one per bucket, and the stored Q is replaced by its bucket's
   * representative. When @c bulk_attenuation is false kappa is neither
   * shifted nor relaxed. No-op when there are no attenuating elements.
   * Host-only; const because only view data is mutated.
   *
   * The shift itself depends on the property tag (see
   * IsotropicModulusShift and AnisotropicModulusShift); anisotropic shear is
   * shifted in the point's radial frame: Cartesian z, the local vertical of
   * globe mantle elements, or globe's inner-core update.
   *
   * @throws std::runtime_error if called a second time, since the moduli in
   *         @p props would then be shifted twice, or if globe anisotropic
   *         elements need spherical coordinates that were not provided.
   *
   * @tparam PropsContainer Property container: h_kappa/h_mu (isotropic) or
   *                        h_c11..h_c66 (anisotropic), and element_range.
   * @tparam SphericalView Host spherical coordinate cache type.
   * @tparam ElementContext Callable returning AttenuationElementContext.
   * @param props The just-read property container
   * @param fc Band-center frequency for modulus scaling
   * @param f0 Reference frequency
   * @param band Attenuation frequency band
   * @param tau_sigma Stress relaxation times
   * @param spherical Setup-only spherical coordinate cache (globe mantle
   * anisotropy only).
   * @param context Global element index to region/radial-zone metadata.
   * @param tables Q tables shared with the other containers of this setup,
   * so all containers use the same representative Q per bucket.
   */
  template <typename PropsContainer, typename SphericalView,
            typename ElementContext>
  void
  recompute(const PropsContainer &props, const specfem::units::Hertz fc,
            const specfem::units::Hertz f0,
            const specfem::utilities::Band<specfem::units::Hertz> &band,
            const Kokkos::View<type_real[N_SLS],
                               Kokkos::DefaultHostExecutionSpace> &tau_sigma,
            const SphericalView &spherical, const ElementContext &context,
            specfem::attenuation::globe::QualityFactorTables<N_SLS> &tables)
      const {
    if (element_range.size() == 0) {
      return;
    }
    if (h_recomputed()) {
      throw std::runtime_error(
          "attenuation_medium::recompute() called twice: the moduli are "
          "already unrelaxed and would be scaled a second time.");
    }
    const auto shift = modulus_shift(
        props, spherical,
        element_range.begin_index() - props.element_range.begin_index());
    if (shift.needs_spherical_coordinates && spherical.data() == nullptr) {
      for (int i = 0; i < element_range.size(); ++i) {
        if (context(element_range(i)).needs_spherical_coordinates()) {
          throw std::runtime_error(
              "Anisotropic attenuation requires spherical coordinates");
        }
      }
    }
    h_recomputed() = true;

    using Fit = specfem::attenuation::globe::QualityFactorFit<N_SLS>;
    const auto fit = [&](const type_real q) {
      return specfem::attenuation::globe::fit_quality_factor<N_SLS>(
          q, tau_sigma, band, fc.raw(), f0.raw());
    };
    auto &bulk_table = tables.bulk;
    auto &shear_table = tables.shear;
    const int fits_before = tables.fits();
    long long points = 0;
    // Without bulk attenuation kappa keeps scale 1 and zero relaxation rates.
    Fit no_bulk;
    no_bulk.properties.beta = decltype(no_bulk.properties.beta)("beta");
    no_bulk.properties.one_minus_sum_beta = 1;

    const int l_ngllz = h_Qkappa.extent(1);
    const int l_nglly = h_Qkappa.extent(2);
    const int l_ngllx = h_Qkappa.extent(3);
    for (int i = 0; i < element_range.size(); ++i) {
      const int ispec = element_range(i);
      const auto metadata = context(ispec);
      if (metadata.fictitious())
        continue;
      for (int iz = 0; iz < l_ngllz; ++iz) {
        for (int iy = 0; iy < l_nglly; ++iy) {
          for (int ix = 0; ix < l_ngllx; ++ix) {
            const Fit *kappa_fit = &no_bulk;
            if (bulk_attenuation) {
              const type_real q =
                  bulk_table.representative(h_Qkappa(i, iz, iy, ix));
              h_Qkappa(i, iz, iy, ix) = q;
              kappa_fit = &bulk_table.fit(q, fit);
            }
            const type_real q =
                shear_table.representative(h_Qmu(i, iz, iy, ix));
            h_Qmu(i, iz, iy, ix) = q;
            const Fit &mu_fit = shear_table.fit(q, fit);
            const auto moduli =
                shift(i, ispec, iz, iy, ix, metadata, *kappa_fit, mu_fit);
            set_relaxation_rates(i, iz, iy, ix, moduli, kappa_fit->properties,
                                 mu_fit.properties, tau_sigma);
            ++points;
          }
        }
      }
    }
    Kokkos::deep_copy(kappa_relaxation_rate, h_kappa_relaxation_rate);
    Kokkos::deep_copy(mu_relaxation_rate, h_mu_relaxation_rate);
    // The fit count is bounded by occupied Q buckets, not by points; a count
    // near the point count means the table is not being hit.
    specfem::Logger::debug(
        [&](std::ostringstream &message) {
          message << "Attenuation recompute ("
                  << specfem::element::to_string(PropertyTag)
                  << "): " << tables.fits() - fits_before << " Q fits for "
                  << points << " GLL points";
        },
        false);
  }

  /**
   * @brief recompute() with Q tables private to this call.
   *
   * For a container recomputed on its own, e.g. after a model read.
   */
  template <typename PropsContainer, typename SphericalView,
            typename ElementContext>
  void
  recompute(const PropsContainer &props, const specfem::units::Hertz fc,
            const specfem::units::Hertz f0,
            const specfem::utilities::Band<specfem::units::Hertz> &band,
            const Kokkos::View<type_real[N_SLS],
                               Kokkos::DefaultHostExecutionSpace> &tau_sigma,
            const SphericalView &spherical,
            const ElementContext &context) const {
    specfem::attenuation::globe::QualityFactorTables<N_SLS> tables;
    recompute(props, fc, f0, band, tau_sigma, spherical, context, tables);
  }

  /**
   * @brief Relaxation rates of one GLL point from its unrelaxed moduli.
   *
   * \f$\kappa\,\beta^\kappa_j/(\tau_{\sigma_j}\Sigma^\kappa)\f$ and
   * \f$2\mu\,\beta^\mu_j/(\tau_{\sigma_j}\Sigma^\mu)\f$, with
   * \f$\Sigma\f$ the one_minus_sum_beta of each fit; matches SPECFEM3D's
   * factor_common scaled by the unrelaxed modulus.
   *
   * @param i Attenuation-local element index
   * @param iz, iy, ix GLL indices
   * @param moduli Unrelaxed moduli of the point
   * @param kappa Bulk relaxation properties (zero defects when not attenuated)
   * @param mu Shear relaxation properties
   * @param tau_sigma Stress relaxation times
   */
  void set_relaxation_rates(
      const int i, const int iz, const int iy, const int ix,
      const UnrelaxedModuli &moduli,
      const specfem::attenuation::AttenuationPropertyValues<N_SLS> &kappa,
      const specfem::attenuation::AttenuationPropertyValues<N_SLS> &mu,
      const Kokkos::View<type_real[N_SLS], Kokkos::DefaultHostExecutionSpace>
          &tau_sigma) const {
    for (int j = 0; j < N_SLS; ++j) {
      const type_real tauinv_j = 1.0 / tau_sigma(j);
      h_kappa_relaxation_rate(i, iz, iy, ix, j) =
          moduli.kappa * kappa.beta(j) * tauinv_j / kappa.one_minus_sum_beta;
      h_mu_relaxation_rate(i, iz, iy, ix, j) =
          moduli.mu * 2.0 * mu.beta(j) * tauinv_j / mu.one_minus_sum_beta;
    }
  }

  /**
   * @brief Per-point modulus shift for isotropic properties.
   * @param props Property container exposing h_kappa and h_mu
   * @param offset Attenuation-to-property element index offset
   * @return IsotropicModulusShift writing into @p props and this container
   */
  template <typename PropsContainer, typename SphericalView>
  auto modulus_shift(const PropsContainer &props, const SphericalView &,
                     const int offset) const
    requires(PropertyTag == specfem::element::property_tag::isotropic)
  {
    return IsotropicModulusShift<PropsContainer, q_view_type>{
      props, h_kappa_scale, h_mu_scale, offset
    };
  }

  /**
   * @brief Per-point modulus shift for anisotropic properties.
   * @param props Property container exposing h_c11..h_c66
   * @param spherical Host spherical coordinate cache
   * @param offset Attenuation-to-property element index offset
   * @return AnisotropicModulusShift writing into @p props and this container
   */
  template <typename PropsContainer, typename SphericalView>
  auto modulus_shift(const PropsContainer &props,
                     const SphericalView &spherical, const int offset) const
    requires(PropertyTag == specfem::element::property_tag::anisotropic)
  {
    using CijView = std::remove_cvref_t<decltype(props.h_c11)>;
    return AnisotropicModulusShift<CijView, q_view_type, correction_view_type,
                                   SphericalView>{
      { props.h_c11, props.h_c12, props.h_c13, props.h_c14, props.h_c15,
        props.h_c16, props.h_c22, props.h_c23, props.h_c24, props.h_c25,
        props.h_c26, props.h_c33, props.h_c34, props.h_c35, props.h_c36,
        props.h_c44, props.h_c45, props.h_c46, props.h_c55, props.h_c56,
        props.h_c66 },
      h_kappa_scale,
      h_mu_scale,
      h_cij_correction,
      spherical,
      offset
    };
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
