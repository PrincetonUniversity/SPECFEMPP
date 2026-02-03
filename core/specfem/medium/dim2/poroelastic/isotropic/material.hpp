
#pragma once

#include "enumerations/medium.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium_container {

namespace impl {

template <specfem::dimension::type DimensionTag>
struct AttenuationValues<
    DimensionTag, specfem::element::medium_tag::poroelastic,
    specfem::element::attenuation_tag::constant_isotropic> {
public:
  type_real Qmu; ///< Attenuation factor for shear modulus
  AttenuationValues(const type_real &Qmu) : Qmu(Qmu) {
    if (this->Qmu <= 0.0) {
      throw std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  };

  bool operator==(const AttenuationValues &other) const {
    return (std::abs(this->Qmu - other.Qmu) < 1e-6);
  }

  std::string print() const {
    std::ostringstream message;

    message << "      Qmu : " << this->Qmu << "\n";

    return message.str();
  }
};

} // namespace impl

/**
 * @brief Template specialization for poroelastic isotropic material
 * properties
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::attenuation_tag AttenuationTag>
class material<DimensionTag, specfem::element::medium_tag::poroelastic,
               specfem::element::property_tag::isotropic, AttenuationTag>
    : impl::AttenuationValues<DimensionTag,
                              specfem::element::medium_tag::poroelastic,
                              AttenuationTag> {
public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::poroelastic; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;          ///< Property tag
  constexpr static auto attenuation_tag = AttenuationTag; ///< Attenuation tag

  using attenuation = impl::AttenuationValues<
      DimensionTag, specfem::element::medium_tag::poroelastic, AttenuationTag>;

  /**
   * @name Constructors
   */
  ///@{

  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<T == specfem::element::attenuation_tag::none, int> = 0>
  material(type_real rhos, type_real rhof, type_real phi, type_real tortuosity,
           type_real kxx, type_real kxz, type_real kzz, type_real Ks,
           type_real Kf, type_real Kfr, type_real etaf, type_real mufr)
      : rhos(rhos), rhof(rhof), phi(phi), tortuosity(tortuosity), kxx(kxx),
        kxz(kxz), kzz(kzz), Ks(Ks), Kf(Kf), Kfr(Kfr), etaf(etaf), mufr(mufr) {}

  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<
          T == specfem::element::attenuation_tag::constant_isotropic, int> = 0>
  material(type_real rhos, type_real rhof, type_real phi, type_real tortuosity,
           type_real kxx, type_real kxz, type_real kzz, type_real Ks,
           type_real Kf, type_real Kfr, type_real etaf, type_real mufr,
           type_real Qmu)
      : attenuation(Qmu), rhos(rhos), rhof(rhof), phi(phi),
        tortuosity(tortuosity), kxx(kxx), kxz(kxz), kzz(kzz), Ks(Ks), Kf(Kf),
        Kfr(Kfr), etaf(etaf), mufr(mufr) {}
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
  bool operator==(
      const material<dimension_tag, specfem::element::medium_tag::poroelastic,
                     specfem::element::property_tag::isotropic, attenuation_tag>
          &other) const {
    return (std::abs(this->rhos - other.rhos) < 1e-6 &&
            std::abs(this->rhof - other.rhof) < 1e-6 &&
            std::abs(this->phi - other.phi) < 1e-6 &&
            std::abs(this->tortuosity - other.tortuosity) < 1e-6 &&
            std::abs(this->kxx - other.kxx) < 1e-6 &&
            std::abs(this->kxz - other.kxz) < 1e-6 &&
            std::abs(this->kzz - other.kzz) < 1e-6 &&
            std::abs(this->Ks - other.Ks) < 1e-6 &&
            std::abs(this->Kf - other.Kf) < 1e-6 &&
            std::abs(this->Kfr - other.Kfr) < 1e-6 &&
            std::abs(this->etaf - other.etaf) < 1e-6 &&
            std::abs(this->mufr - other.mufr) < 1e-6 &&
            attenuation::operator==(other));
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(
      const material<dimension_tag, specfem::element::medium_tag::poroelastic,
                     specfem::element::property_tag::isotropic, attenuation_tag>
          &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension_tag, medium_tag, property_tag,
                                    false>
  get_properties() const {

    const type_real phi = this->phi;
    const type_real rho_s = this->rhos;
    const type_real rho_f = this->rhof;
    const type_real tortuosity = this->tortuosity;
    const type_real mu_G = this->mufr;
    const auto [H_Biot, C_Biot, M_Biot] = this->compute_biot_coefficients();
    const type_real permxx = this->kxx;
    const type_real permxz = this->kxz;
    const type_real permzz = this->kzz;
    const type_real eta_f = this->etaf;
    return { phi,    rho_s,  rho_f,  tortuosity, mu_G,   H_Biot,
             C_Biot, M_Biot, permxx, permxz,     permzz, eta_f };
  }

  /**
   * @brief Print the material properties
   *
   * @return std::string Formatted material properties
   */
  inline std::string print() const {
    std::ostringstream message;

    message << "- Poroelastic Material : \n"
            << "    Properties:\n"
            << "      rhos : " << this->rhos << "\n"
            << "      rhof : " << this->rhof << "\n"
            << "      phi : " << this->phi << "\n"
            << "      tortuosity : " << this->tortuosity << "\n"
            << "      kxx : " << this->kxx << "\n"
            << "      kxz : " << this->kxz << "\n"
            << "      kzz : " << this->kzz << "\n"
            << "      Ks : " << this->Ks << "\n"
            << "      Kf : " << this->Kf << "\n"
            << "      Kfr : " << this->Kfr << "\n"
            << "      etaf : " << this->etaf << "\n"
            << "      mufr : " << this->mufr << "\n"
            << static_cast<const attenuation &>(*this).print();

    return message.str();
  }

private:
  type_real rhos;       ///< Density of the solid phase
  type_real rhof;       ///< Density of the fluid phase
  type_real phi;        ///< Porosity
  type_real tortuosity; ///< Tortuosity of the solid phase
  type_real kxx;        ///< Permeability tensor XX component
  type_real kxz;        ///< Permeability tensor XZ component
  type_real kzz;        ///< Permeability tensor ZZ component
  type_real Ks;         ///< Bulk modulus of the solid phase
  type_real Kf;         ///< Bulk modulus of the fluid phase
  type_real Kfr;        ///< Bulk modulus of the frame
  type_real etaf;       ///< Viscosity of the fluid phase
  type_real mufr;       ///< Shear modulus of the frame

  std::tuple<type_real, type_real, type_real>
  compute_biot_coefficients() const {

    const type_real kappa_s = this->Ks; ///< Bulk modulus of the solid phase
    const type_real kappa_f = this->Kf; ///< Bulk modulus of the fluid phase
    const type_real phi = this->phi;    ///< Porosity

    const type_real kappa_fr = this->Kfr; ///< Bulk modulus of the frame
    const type_real mu_fr = this->mufr;   ///< Shear modulus of the frame

    const type_real D_Biot = kappa_s * (1.0 + phi * (kappa_s / kappa_f - 1.0));

    const type_real fac_inv = 1.0 / (D_Biot - kappa_fr); ///< Helper factor

    const type_real H_Biot =
        (kappa_s - kappa_fr) * (kappa_s - kappa_fr) * fac_inv + kappa_fr +
        4.0 / 3.0 * mu_fr;

    const type_real C_biot = kappa_s * (kappa_s - kappa_fr) * fac_inv;
    const type_real M_biot = kappa_s * kappa_s * fac_inv;

    return std::make_tuple(H_Biot, C_biot, M_biot);
  }
};

} // namespace medium_container
} // namespace specfem
