
#pragma once

#include "enumerations/medium.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium {

/**
 * @brief Template specialization for electromagnetic isotropic (SV) material
 * properties
 *
 */
template <>
class material<specfem::element::medium_tag::electromagnetic_sv,
               specfem::element::property_tag::isotropic> {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::electromagnetic_sv; ///< Medium tag
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
   * @param e0 Effective permittivity in farad per meter
   * @param e11 e11 of effective permittivity in farad per meter
   * @param e33 e33 of effective permittivity in farad per meter
   * @param sig11 sig11 of effective conductivity in siemens per meter
   * @param sig33 sig33 of effective conductivity in siemens per meter
   * @param Qe11 Quality factor of e11 for attenuation
   * @param Qe33 Quality factor of e33 for attenuation
   * @param Qs11 Quality factor of sig11 for attenuation
   * @param Qs33 Quality factor of sig33 for attenuation
   */
  material(type_real mu0, type_real e0, type_real e11, type_real e33,
           type_real sig11, type_real sig33, type_real Qe11, type_real Qe33,
           type_real Qs11, type_real Qs33)
      : mu0(mu0), e0(e0), e11(e11), e33(e33), sig11(sig11), sig33(sig33),
        Qe11(Qe11), Qe33(Qe33), Qs11(Qs11), Qs33(Qs33) {
          // TODO: Add checks for the material properties electromagnetic_sv
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
      const material<specfem::element::medium_tag::electromagnetic_sv,
                     specfem::element::property_tag::isotropic> &other) const {

    return (std::abs(this->mu0 - other.mu0) < 1e-6 &&
            std::abs(this->e0 - other.e0) < 1e-6 &&
            std::abs(this->e11 - other.e11) < 1e-6 &&
            std::abs(this->e33 - other.e33) < 1e-6 &&
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
      const material<specfem::element::medium_tag::electromagnetic_sv,
                     specfem::element::property_tag::isotropic> &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  // TODO: Add the properties for electromagnetic_sv
  // inline specfem::point::properties<dimension, medium_tag, property_tag,
  // false> get_properties() const {
  //   return { this->mu0, this->e0, this->e11, this->e33, this->sig11,
  //   this->sig33,
  //            this->Qe11, this->Qe33, this->Qs11, this->Qs33 };
  // }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Electromagnetic Material : \n"
            << "    Properties:\n"
            << "      magnetic permeability mu0: " << this->mu0 << "\n"
            << "      effective permittivity e0: " << this->e0 << "\n"
            << "      e11 of effective permittivity: " << this->e11 << "\n"
            << "      e33 of effective permittivity: " << this->e33 << "\n"
            << "      sig11 of effective conductivity: " << this->sig11 << "\n"
            << "      sig33 of effective conductivity: " << this->sig33 << "\n"
            << "      Qe11 quality factor of e11: " << this->Qe11 << "\n"
            << "      Qe33 quality factor of e33: " << this->Qe33 << "\n"
            << "      Qs11 quality factor of sig11: " << this->Qs11 << "\n"
            << "      Qs33 quality factor of sig33: " << this->Qs33 << "\n";

    return message.str();
  }

private:
  type_real mu0;   ///< magnetic permeability in henry per meter
  type_real e0;    ///< effective permittivity in farad per meter
  type_real e11;   ///< e11 of effective permittivity in farad per meter
  type_real e33;   ///< e33 of effective permittivity in farad per meter
  type_real sig11; ///< sig11 of effective conductivity in siemens per meter
  type_real sig33; ///< sig33 of effective conductivity in siemens per meter
  type_real Qe11;  ///< quality factor of e11 for attenuation
  type_real Qe33;  ///< quality factor of e33 for attenuation
  type_real Qs11;  ///< quality factor of sig11 for attenuation
  type_real Qs33;  ///< quality factor of sig33 for attenuation
};

} // namespace medium
} // namespace specfem
