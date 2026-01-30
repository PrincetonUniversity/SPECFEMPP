
#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_material_dim2_electromagnetic_isotropic 2D
 * Electromagnetic Isotropic
 *
 */

/**
 * @ingroup specfem_medium_material_dim2_electromagnetic_isotropic
 * @brief Template specialization for electromagnetic isotropic (TE) material
 * properties
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class material<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> > {
public:
  constexpr static auto dimension_tag =
      DimensionTag;                             ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new elastic isotropic material
   *
   * @param mu0 Magnetic permeability in henry per meter
   * @param e0 @f$ \epsilon_{0} @f$ Effective permittivity in farad per meter
   * @param e11_e0 @f$ \epsilon_{11}/\epsilon_{0} @f$ component of the
   * permittivity tensor
   * @param e33_e0 @f$ \epsilon_{33}/\epsilon_{0} @f$ component of the
   * permittivity tensor
   * @param sig11 @f$ \sigma_{11} @f$ component of the conductivity tensor
   * @param sig33 @f$ \sigma_{33} @f$ component of the conductivity tensor
   * @param Qe11 Quality factor of @f$ \epsilon_{11}/\epsilon_{0} @f$ for
   * attenuation
   * @param Qe33 Quality factor of @f$ \epsilon_{33}/\epsilon_{0} @f$ for
   * attenuation
   * @param Qs11 Quality factor of @f$ \sigma_{11} @f$ for attenuation
   * @param Qs33 Quality factor of @f$ \sigma_{33} @f$ for attenuation
   */
  material(type_real mu0, type_real e0, type_real e11_e0, type_real e33_e0,
           type_real sig11, type_real sig33, type_real Qe11, type_real Qe33,
           type_real Qs11, type_real Qs33)
      : mu0(mu0), e0(e0), e11_e0(e11_e0), e33_e0(e33_e0), sig11(sig11),
        sig33(sig33), Qe11(Qe11), Qe33(Qe33), Qs11(Qs11), Qs33(Qs33) {
          // TODO: Add checks for the material properties electromagnetic_te
          // Currently, we there aren't any checks for the material properties
          // but we can add them here
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
  bool operator==(
      const material<dimension_tag, medium_tag,
                     specfem::element::property_tag::isotropic> &other) const {

    return (std::abs(this->mu0 - other.mu0) < 1e-6 &&
            std::abs(this->e0 - other.e0) < 1e-6 &&
            std::abs(this->e11_e0 - other.e11_e0) < 1e-6 &&
            std::abs(this->e33_e0 - other.e33_e0) < 1e-6 &&
            std::abs(this->sig11 - other.sig11) < 1e-6 &&
            std::abs(this->sig33 - other.sig33) < 1e-6 &&
            std::abs(this->Qe11 - other.Qe11) < 1e-6 &&
            std::abs(this->Qe33 - other.Qe33) < 1e-6 &&
            std::abs(this->Qs11 - other.Qs11) < 1e-6 &&
            std::abs(this->Qs33 - other.Qs33) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(
      const material<dimension_tag,
                     specfem::element::medium_tag::electromagnetic_te,
                     specfem::element::property_tag::isotropic> &other) const {
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
    return { static_cast<type_real>(1.0) / this->mu0, this->e0 * this->e11_e0,
             this->e0 * this->e33_e0, this->sig11, this->sig33 };
  }

  /**
   * @brief Print the material properties
   *
   * @return std::string Formatted material properties
   */
  inline std::string print() const {
    std::ostringstream message;

    message << "- Electromagnetic Material : \n"
            << "    Properties:\n"
            << "      magnetic permeability mu0: " << this->mu0 << "\n"
            << "      effective permittivity e0: " << this->e0 << "\n"
            << "      e11 of effective permittivity: " << this->e11_e0 << "\n"
            << "      e33 of effective permittivity: " << this->e33_e0 << "\n"
            << "      sig11 of effective conductivity: " << this->sig11 << "\n"
            << "      sig33 of effective conductivity: " << this->sig33 << "\n"
            << "      Qe11 quality factor of e11: " << this->Qe11 << "\n"
            << "      Qe33 quality factor of e33: " << this->Qe33 << "\n"
            << "      Qs11 quality factor of sig11: " << this->Qs11 << "\n"
            << "      Qs33 quality factor of sig33: " << this->Qs33 << "\n";

    return message.str();
  }

private:
  type_real mu0; ///< @f$ \mu_{0} @f$ Magnetic permeability in henry per meter
  type_real e0;  ///< @f$ \epsilon_{0} @f$ Effective permittivity in farad per
                 ///< meter
  type_real e11_e0; ///< @f$ \epsilon_{11}/\epsilon_{0} @f$ component of the
                    ///< permittivity tensor
  type_real e33_e0; ///< @f$ \epsilon_{33}/\epsilon_{0} @f$ component of the
                    ///< permittivity tensor
  type_real sig11; ///< @f$ \sigma_{11} @f$ component of the conductivity tensor
  type_real sig33; ///< @f$ \sigma_{33} @f$ component of the conductivity tensor
  type_real Qe11;  ///< Quality factor of @f$ \epsilon_{11}/\epsilon_{0} @f$ for
                   ///< attenuation
  type_real Qe33;  ///< Quality factor of @f$ \epsilon_{33}/\epsilon_{0} @f$ for
                   ///< attenuation
  type_real Qs11;  ///< Quality factor of @f$ \sigma_{11} @f$ for attenuation
  type_real Qs33;  ///< Quality factor of @f$ \sigma_{33} @f$ for attenuation
};

} // namespace medium
} // namespace specfem
