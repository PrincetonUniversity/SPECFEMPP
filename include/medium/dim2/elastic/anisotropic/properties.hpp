#pragma once

#include "enumerations/specfem_enums.hpp"
#include "medium/properties.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium {

/**
 * @brief Template specialization for elastic anisotropic material properties
 *
 */
template <>
class properties<specfem::element::medium_tag::elastic_sv,
                 specfem::element::property_tag::anisotropic> {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::elastic_sv; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new elastic anisotropic material
   * @param density Density of the material
   * @param c11 Elastic constant
   * @param c13 Elastic constant
   * @param c15 Elastic constant
   * @param c33 Elastic constant
   * @param c35 Elastic constant
   * @param c55 Elastic constant
   * @param c12 Elastic constant
   * @param c23 Elastic constant
   * @param c25 Elastic constant
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   */
  properties(const type_real &density, const type_real &c11,
             const type_real &c13, const type_real &c15, const type_real &c33,
             const type_real &c35, const type_real &c55, const type_real &c12,
             const type_real &c23, const type_real &c25,
             const type_real &Qkappa, const type_real &Qmu)
      : density(density), c11(c11), c13(c13), c15(c15), c33(c33), c35(c35),
        c55(c55), c12(c12), c23(c23), c25(c25), Qkappa(Qkappa), Qmu(Qmu) {

    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }

    /// @todo Add checks for the elastic constants
  };

  /**
   * @brief Default constructor
   *
   */
  properties() = default;

  ///@}

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(const properties<specfem::element::medium_tag::elastic_sv,
                                   specfem::element::property_tag::anisotropic>
                      &other) const {
    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->c11 - other.c11) < 1e-6 &&
            std::abs(this->c13 - other.c13) < 1e-6 &&
            std::abs(this->c15 - other.c15) < 1e-6 &&
            std::abs(this->c33 - other.c33) < 1e-6 &&
            std::abs(this->c35 - other.c35) < 1e-6 &&
            std::abs(this->c55 - other.c55) < 1e-6 &&
            std::abs(this->c12 - other.c12) < 1e-6 &&
            std::abs(this->c23 - other.c23) < 1e-6 &&
            std::abs(this->c25 - other.c25) < 1e-6 &&
            std::abs(this->Qkappa - other.Qkappa) < 1e-6 &&
            std::abs(this->Qmu - other.Qmu) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(const properties<specfem::element::medium_tag::elastic_sv,
                                   specfem::element::property_tag::anisotropic>
                      &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension, medium_tag, property_tag, false>
  get_properties() const {
    return { c11, c13, c15, c33, c35, c55, c12, c23, c25, density };
  }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Elastic Anisotropic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      c11 : " << this->c11 << "\n"
            << "      c13 : " << this->c13 << "\n"
            << "      c15 : " << this->c15 << "\n"
            << "      c33 : " << this->c33 << "\n"
            << "      c35 : " << this->c35 << "\n"
            << "      c55 : " << this->c55 << "\n"
            << "      c12 : " << this->c12 << "\n"
            << "      c23 : " << this->c23 << "\n"
            << "      c25 : " << this->c25 << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      Qmu : " << this->Qmu << "\n";
    return message.str();
  }

private:
  type_real density; ///< Density of the material
  type_real c11;     ///< Elastic constant
  type_real c13;     ///< Elastic constant
  type_real c15;     ///< Elastic constant
  type_real c33;     ///< Elastic constant
  type_real c35;     ///< Elastic constant
  type_real c55;     ///< Elastic constant
  type_real c12;     ///< Elastic constant
  type_real c23;     ///< Elastic constant
  type_real c25;     ///< Elastic constant
  type_real Qkappa;  ///< Attenuation factor for bulk modulus
  type_real Qmu;     ///< Attenuation factor for shear modulus
};

} // namespace medium
} // namespace specfem
