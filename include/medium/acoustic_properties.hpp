#pragma once

#include "enumerations/specfem_enums.hpp"
#include "point/properties.hpp"
#include "properties.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium {

/**
 * @brief Template specialization for acoustic isotropic material properties
 *
 */
template <>
class properties<specfem::element::medium_tag::acoustic,
                 specfem::element::property_tag::isotropic> {

public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::acoustic; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new acoustic isotropic material
   *
   * @param density Density of the material
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  properties(const type_real &density, const type_real &cp,
             const type_real &Qkappa, const type_real &Qmu,
             const type_real &compaction_grad)
      : density(density), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        lambda(lambdaplus2mu), kappa(lambda) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  }

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(const properties<specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            std::abs(this->Qkappa - other.Qkappa) < 1e-6 &&
            std::abs(this->Qmu - other.Qmu) < 1e-6 &&
            std::abs(this->compaction_grad - other.compaction_grad) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(const properties<specfem::element::medium_tag::acoustic,
                                   specfem::element::property_tag::isotropic>
                      &other) const {
    return !(*this == other);
  }

  /**
   * @brief Default constructor
   *
   */
  properties() = default;
  ///@}

  ~properties() = default;

  /**
   * @brief Get the properties of the material
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension, medium_tag, property_tag, false>
  get_properties() const {
    return { static_cast<type_real>(1.0) /
                 static_cast<type_real>(lambdaplus2mu),
             static_cast<type_real>(1.0) / static_cast<type_real>(density),
             this->kappa };
  }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Acoustic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      lambda : " << this->lambda << "\n"
            << "      youngs modulus : 0.0 \n"
            << "      poisson ratio :  0.5 \n";

    return message.str();
  }

private:
  type_real density;         ///< Density of the material
  type_real cp;              ///< Compressional wave speed
  type_real Qkappa;          ///< Attenuation factor for bulk modulus
  type_real Qmu;             ///< Attenuation factor for shear modulus
  type_real compaction_grad; ///< Compaction gradient
  type_real lambdaplus2mu;   ///< Lame parameter
  type_real lambda;          ///< Lame parameter
  type_real kappa;           ///< Bulk modulus
};

} // namespace medium
} // namespace specfem
