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
 * @see specfem::dimension::type::dim2
 * @see specfem::medium::material
 *
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
struct material<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
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
   * @param density Density of the material
   * @param cs Shear wave speed
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  material(const type_real &density, const type_real &cs, const type_real &cp,
           const type_real &Qkappa, const type_real &Qmu,
           const type_real &compaction_grad)
      : density(density), cs(cs), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        mu(density * cs * cs), lambda(lambdaplus2mu - 2.0 * mu),

        kappa(density * (cp * cp - (4.0 / 3.0) * cs * cs)),
        young(9.0 * kappa * mu / (3.0 * kappa + mu)),
        poisson(0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs)) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }

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
  bool operator==(
      const material<dimension_tag, medium_tag,
                     specfem::element::property_tag::isotropic> &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            std::abs(this->cs - other.cs) < 1e-6 &&
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
  bool operator!=(
      const material<dimension_tag, medium_tag,
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
    return { this->kappa, this->mu, this->density };
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
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      Qmu : " << this->Qmu << "\n"
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
  type_real Qkappa;          ///< Attenuation factor for bulk modulus
  type_real Qmu;             ///< Attenuation factor for shear modulus
  type_real compaction_grad; ///< Compaction gradient
  type_real lambdaplus2mu;   ///< Lambda plus 2*mu (P-wave modulus)
  type_real mu;              ///< Lame parameter
  type_real lambda;          ///< Lame parameter
  type_real kappa;           ///< Bulk modulus
  type_real young;           ///< Young's modulus
  type_real poisson;         ///< Poisson's ratio
};

} // namespace medium
} // namespace specfem
