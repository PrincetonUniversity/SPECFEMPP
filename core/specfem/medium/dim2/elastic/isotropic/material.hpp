#pragma once

#include "specfem/attenuation.hpp"
#include "specfem/constants.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium_container {

namespace impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct ComputedAttenuationValues<
    DimensionTag, MediumTag,
    specfem::element::attenuation_tag::constant_isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {

  using view_type = Kokkos::View<type_real[specfem::constants::N_SLS],
                                 Kokkos::LayoutRight, Kokkos::HostSpace>;

  type_real kappa_scale; ///< Scaling factor for bulk modulus attenuation
  type_real mu_scale;    ///< Scaling factor for shear modulus attenuation

  view_type tau_epsilon_kappa; ///< Relaxation times for SLS mechanisms
  view_type tau_epsilon_mu;    ///< Relaxation times for SLS mechanisms
  specfem::attenuation::AttenuationPropertyValues<specfem::constants::N_SLS>
      kappa_attenuation_properties; ///< Struct to hold computed attenuation
                                    ///< properties
  specfem::attenuation::AttenuationPropertyValues<specfem::constants::N_SLS>
      mu_attenuation_properties; ///< Struct to hold computed attenuation
                                 ///< properties

public:
  ComputedAttenuationValues() = default;

  ComputedAttenuationValues(
      const type_real &Qkappa, const type_real &Qmu, const type_real &f0,
      const type_real &fc,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const view_type &tau_sigma) {

    this->tau_epsilon_kappa =
        specfem::attenuation::compute_tau_eps<specfem::constants::N_SLS>(
            Qkappa, tau_sigma, band);

    this->tau_epsilon_mu =
        specfem::attenuation::compute_tau_eps<specfem::constants::N_SLS>(
            Qmu, tau_sigma, band);

    this->kappa_attenuation_properties =
        specfem::attenuation::get_attenuation_property_values<
            specfem::constants::N_SLS>(tau_sigma, tau_epsilon_kappa);

    this->mu_attenuation_properties =
        specfem::attenuation::get_attenuation_property_values<
            specfem::constants::N_SLS>(tau_sigma, tau_epsilon_mu);

    this->kappa_scale = specfem::attenuation::get_attenuation_scale_factor<
        specfem::constants::N_SLS>(fc, tau_epsilon_kappa, tau_sigma, Qkappa,
                                   f0);
    this->mu_scale = specfem::attenuation::get_attenuation_scale_factor<
        specfem::constants::N_SLS>(fc, tau_epsilon_mu, tau_sigma, Qmu, f0);
  }
};

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct AttenuationValues<
    DimensionTag, MediumTag,
    specfem::element::attenuation_tag::constant_isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
public:
  type_real Qkappa; ///< Attenuation factor for bulk modulus
  type_real Qmu;    ///< Attenuation factor for shear modulus

  constexpr static auto dimension_tag =
      DimensionTag;                             ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic; ///< Attenuation
                                                             ///< tag
private:
  ComputedAttenuationValues<dimension_tag, medium_tag, attenuation_tag>
      computed_values; ///< Struct to hold computed attenuation properties

  bool compute_properties_called =
      false; ///< Flag to check if properties have been computed

  using view_type = Kokkos::View<type_real[specfem::constants::N_SLS],
                                 Kokkos::LayoutRight, Kokkos::HostSpace>;

public:
  AttenuationValues() = default;

  AttenuationValues(const type_real &Qkappa, const type_real &Qmu)
      : Qkappa(Qkappa), Qmu(Qmu) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      throw std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  }

  bool operator==(const AttenuationValues &other) const {
    return (std::abs(this->Qkappa - other.Qkappa) < 1e-6 &&
            std::abs(this->Qmu - other.Qmu) < 1e-6);
  }

  const ComputedAttenuationValues<dimension_tag, medium_tag, attenuation_tag> &
  compute_attenuation_properties(
      const type_real &f0, const type_real &fc,
      const specfem::utilities::Band<specfem::units::Hertz> &band,
      const view_type &tau_sigma) {

    if (this->compute_properties_called) {
      return this->computed_values;
    }

    this->computed_values =
        ComputedAttenuationValues<dimension_tag, medium_tag, attenuation_tag>(
            this->Qkappa, this->Qmu, f0, fc, band, tau_sigma);

    this->compute_properties_called = true;

    return this->computed_values;
  }

  // Getters for computed properties with error handling
  const ComputedAttenuationValues<dimension_tag, medium_tag, attenuation_tag> &
  get_computed_values() const {
    if (!this->compute_properties_called) {
      throw std::runtime_error(
          "Attenuation properties have not been computed yet. Call "
          "compute_attenuation_properties() first.");
    }
    return this->computed_values;
  }

  std::string print() const {
    std::ostringstream message;

    message << "      Qkappa : " << this->Qkappa << "\n"
            << "      Qmu : " << this->Qmu << "\n";
    if (this->compute_properties_called) {
      message << "      Computed attenuation properties:\n"
              << "        kappa scale factor: "
              << this->computed_values.kappa_scale << "\n"
              << "        mu scale factor:    "
              << this->computed_values.mu_scale << "\n";
    } else {
      message << "      Attenuation properties have not been computed yet.\n";
    }

    return message.str();
  }
};

} // namespace impl

/**
 * @defgroup specfem_medium_material_dim2_elastic_isotropic 2D Elastic Isotropic
 * Material
 */

/**
 * @ingroup specfem_medium_material_dim2_elastic_isotropic
 * @brief Material specialization for 2D elastic isotropic media
 *
 * This struct holds the properties of an elastic isotropic material in 2D
 * space. It includes the density, shear wave speed, compressional wave speed,
 * attenuation factors, and compaction gradient. The struct also provides
 *
 * @tparam MediumTag The medium tag that must satisfy elastic medium properties
 * @tparam PropertyTag The property tag that must be isotropic
 * @tparam Enable The enable_if condition that must be satisfied
 *
 * @see specfem::element::is_elastic
 * @see specfem::element::dimension_tag::dim2
 * @see specfem::medium_container::material
 *
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag>
struct material<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic,
    AttenuationTag,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : impl::AttenuationValues<DimensionTag, MediumTag, AttenuationTag> {
public:
  constexpr static auto dimension_tag =
      DimensionTag;                             ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;          ///< Property tag
  constexpr static auto attenuation_tag = AttenuationTag; ///< Attenuation tag

  using attenuation =
      impl::AttenuationValues<DimensionTag, MediumTag, AttenuationTag>;

  /**
   * @name Constructors
   */
  ///@{
  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<T == specfem::element::attenuation_tag::none, int> = 0>
  material(const type_real &density, const type_real &cs, const type_real &cp,
           const type_real &compaction_grad)
      : density(density), cs(cs), cp(cp), compaction_grad(compaction_grad),
        lambdaplus2mu(density * cp * cp), mu(density * cs * cs),
        lambda(lambdaplus2mu - 2.0 * mu),
        kappa(density * (cp * cp - (4.0 / 3.0) * cs * cs)),
        young(9.0 * kappa * mu / (3.0 * kappa + mu)),
        poisson(0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs)) {
    if (this->poisson < -1.0 || this->poisson > 0.5)
      std::runtime_error("Poisson's ratio out of range");
  };
  /**
   * @brief Construct a new elastic isotropic material
   *
   * @param density Density of the material
   * @param cs Shear wave speed
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<
          T == specfem::element::attenuation_tag::constant_isotropic, int> = 0>
  material(const type_real &density, const type_real &cs, const type_real &cp,
           const type_real &Qkappa, const type_real &Qmu,
           const type_real &compaction_grad)
      : attenuation(Qkappa, Qmu), density(density), cs(cs), cp(cp),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        mu(density * cs * cs), lambda(lambdaplus2mu - 2.0 * mu),
        kappa(density * (cp * cp - (4.0 / 3.0) * cs * cs)),
        young(9.0 * kappa * mu / (3.0 * kappa + mu)),
        poisson(0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs)) {
    if (this->poisson < -1.0 || this->poisson > 0.5)
      std::runtime_error("Poisson's ratio out of range");
  };
  /**
   * @brief Default constructor
   *
   */
  material() = default;

  ///@}

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(const material<dimension_tag, medium_tag,
                                 specfem::element::property_tag::isotropic,
                                 attenuation_tag> &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            std::abs(this->cs - other.cs) < 1e-6 &&
            attenuation::operator==(other) &&
            std::abs(this->compaction_grad - other.compaction_grad) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(const material<dimension_tag, medium_tag,
                                 specfem::element::property_tag::isotropic,
                                 attenuation_tag> &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<T == specfem::element::attenuation_tag::none, int> = 0>
  inline specfem::point::properties<specfem::tags::Tags<
      dimension_tag, medium_tag, property_tag, attenuation_tag, false> >
  get_properties() const {
    return { this->kappa, this->mu, this->density };
  }

  /**
   * @brief Get the material properties with attenuation scaling
   *
   * @return specfem::point::properties Material properties with attenuation
   * scaling
   */
  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<
          T == specfem::element::attenuation_tag::constant_isotropic, int> = 0>
  inline specfem::point::properties<specfem::tags::Tags<
      dimension_tag, medium_tag, property_tag, attenuation_tag, false> >
  get_properties() const {
    return { this->kappa * attenuation::get_computed_values().kappa_scale,
             this->mu * attenuation::get_computed_values().mu_scale,
             this->density };
  }

  /**
   * @brief Print the material properties
   *
   * @return std::string Formatted material properties
   */
  inline std::string print() const {
    std::ostringstream message;

    message << "- Elastic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cs : " << this->cs << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      mu : " << this->mu << "\n"
            << static_cast<const attenuation &>(*this).print()
            << "      lambda : " << this->lambda << "\n"
            << "      mu : " << this->mu << "\n"
            << "      youngs modulus : " << this->young << "\n"
            << "      poisson ratio : " << this->poisson << "\n";

    return message.str();
  }

protected:
  type_real density;         ///< Density of the material
  type_real cs;              ///< Shear wave speed
  type_real cp;              ///< Compressional wave speed
  type_real compaction_grad; ///< Compaction gradient
  type_real lambdaplus2mu;   ///< Lambda plus 2*mu (P-wave modulus)
  type_real mu;              ///< Lame parameter
  type_real lambda;          ///< Lame parameter
  type_real kappa;           ///< Bulk modulus
  type_real young;           ///< Young's modulus
  type_real poisson;         ///< Poisson's ratio
};

} // namespace medium_container
} // namespace specfem
