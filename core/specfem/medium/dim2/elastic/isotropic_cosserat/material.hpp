
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
 * @defgroup specfem_medium_material_dim2_elastic_isotropic_cosserat 2D Elastic
 * Isotropic Material
 */

/**
 * @ingroup specfem_medium_material_dim2_elastic_isotropic_cosserat
 * properties
 * @brief Material specialization for 2D elastic isotropic cosserat media
 *
 * This struct holds the properties of an elastic isotropic cosserat material in
 * 2D.
 *
 * @tparam MediumTag The medium tag that must satisfy elastic medium properties
 * @tparam PropertyTag The property tag that must be isotropic
 * @tparam Enable The enable_if condition that must be satisfied
 *
 * @see specfem::element::is_elastic
 * @see specfem::dimension::type::dim2
 * @see specfem::medium::material
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class material<
    DimensionTag, MediumTag, specfem::element::property_tag::isotropic_cosserat,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> > {
public:
  constexpr static auto dimension_tag =
      DimensionTag;                             ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic_cosserat; ///< Property tag

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new elastic spin isotropic material
   *
   * @param rho Density @f$ \rho @f$
   * @param kappa Bulk modulus @f$ \kappa @f$
   * @param mu Shear modulus @f$ \mu @f$
   * @param nu Symmetry breaking coupling modulus @f$ \nu @f$
   * @param j Inertia density  @f$ j @f$
   * @param lambda_c Coupling modulus @f$ \lambda_c @f$
   * @param mu_c Coupling shear modulus @f$ \mu_c @f$
   * @param nu_c Coupling symmetry breaking modulus @f$ \nu_c @f$
   *
   */
  material(type_real rho, type_real kappa, type_real mu, type_real nu,
           type_real j, type_real lambda_c, type_real mu_c, type_real nu_c)
      : rho(rho), kappa(kappa), mu(mu), nu(nu), j(j), lambda_c(lambda_c),
        mu_c(mu_c), nu_c(nu_c) {
          // TODO: Add checks for the material properties elastin spin
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
                     specfem::element::property_tag::isotropic_cosserat> &other)
      const {

    return (std::abs(this->rho - other.rho) < 1e-6 &&
            std::abs(this->kappa - other.kappa) < 1e-6 &&
            std::abs(this->mu - other.mu) < 1e-6 &&
            std::abs(this->nu - other.nu) < 1e-6 &&
            std::abs(this->j - other.j) < 1e-6 &&
            std::abs(this->lambda_c - other.lambda_c) < 1e-6 &&
            std::abs(this->mu_c - other.mu_c) < 1e-6 &&
            std::abs(this->nu_c - other.nu_c) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(
      const material<dimension_tag, specfem::element::medium_tag::elastic_psv_t,
                     specfem::element::property_tag::isotropic_cosserat> &other)
      const {
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
    return { rho, kappa, mu, nu, j, lambda_c, mu_c, nu_c };
  }

  /**
   * @brief Print the material properties
   *
   * @return std::string Formatted material properties
   */
  inline std::string print() const {
    std::ostringstream message;

    message << "- Elastic Isotropic Cosserat Material : \n"
            << "    Properties:\n"
            << "      Density:                " << this->rho << "\n"
            << "      Bulk modulus:           " << this->kappa << "\n"
            << "      Shear modulus:          " << this->mu << "\n"
            << "      Coupling modulus:       " << this->nu << "\n"
            << "      Inertia density:        " << this->j << "\n"
            << "      Coupling bulk modulus:  " << this->lambda_c << "\n"
            << "      Coupling shear modulus: " << this->mu_c << "\n"
            << "      Coupling modulus:       " << this->nu_c;

    return message.str();
  }

private:
  type_real rho;      ///< Density @f$ \rho @f$
  type_real kappa;    ///< Bulk modulus @f$ \kappa @f$
  type_real mu;       ///< Shear modulus @f$ \mu @f$
  type_real nu;       ///< Symmetry breaking coupling modulus @f$ \nu @f$
  type_real j;        ///< Intertia density  @f$ j @f$
  type_real lambda_c; ///< Coupling bulk modulus @f$ \lambda_c @f$
  type_real mu_c;     ///< Coupling shear modulus @f$ \mu_c @f$
  type_real nu_c;     ///< Coupling symmetry breaking modulus @f$ \nu_c @f$
};

} // namespace medium
} // namespace specfem
