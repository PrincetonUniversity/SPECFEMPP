
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
 * @brief Template specialization for 2D elastic spin isotropic material
 * properties
 *
 * This class is mainly a container to hold the material properties that are
 * putput by the mesher and read by the mesh reader.
 *
 */
template <specfem::element::medium_tag MediumTag>
class material<
    MediumTag, specfem::element::property_tag::isotropic,
    std::enable_if_t<specfem::element::is_elastic_spin<MediumTag>::value> > {
public:
  constexpr static auto dimension =
      specfem::dimension::type::dim2;           ///< Dimension of the material
  constexpr static auto medium_tag = MediumTag; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic; ///< Property tag

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
   * @param kappa_c Coupling bulk modulus @f$ \kappa_c @f$
   * @param mu_c Coupling shear modulus @f$ \mu_c @f$
   * @param nu_c Coupling symmetry breaking modulus @f$ \nu_c @f$
   *
   */
  material(type_real rho, type_real kappa, type_real mu, type_real nu,
           type_real j, type_real kappa_c, type_real mu_c, type_real nu_c)
      : rho(rho), kappa(kappa), mu(mu), nu(nu), j(j), kappa_c(kappa_c),
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
      const material<MediumTag, specfem::element::property_tag::isotropic>
          &other) const {

    return (std::abs(this->rho - other.rho) < 1e-6 &&
            std::abs(this->kappa - other.kappa) < 1e-6 &&
            std::abs(this->mu - other.mu) < 1e-6 &&
            std::abs(this->nu - other.nu) < 1e-6 &&
            std::abs(this->j - other.j) < 1e-6 &&
            std::abs(this->kappa_c - other.kappa_c) < 1e-6 &&
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
      const material<specfem::element::medium_tag::elastic_psv_t,
                     specfem::element::property_tag::isotropic> &other) const {
    return !(*this == other);
  }

  /**
   * @brief Get the material properties
   *
   * @return specfem::point::properties Material properties
   */
  inline specfem::point::properties<dimension, medium_tag, property_tag, false>
  get_properties() const {
    return {};
  }

  inline std::string print() const {
    std::ostringstream message;

    message << "- Elastic Isotropic Spin Material : \n"
            << "    Properties:\n"
            << "      Density:                " << this->rho << "\n"
            << "      Bulk modulus:           " << this->kappa << "\n"
            << "      Shear modulus:          " << this->mu << "\n"
            << "      Coupling modulus:       " << this->nu << "\n"
            << "      Inertia density:        " << this->j << "\n"
            << "      Coupling bulk modulus:  " << this->kappa_c << "\n"
            << "      Coupling shear modulus: " << this->mu_c << "\n"
            << "      Coupling modulus:       " << this->nu_c;

    return message.str();
  }

private:
  type_real rho;     ///< Density @f$ \rho @f$
  type_real kappa;   ///< Bulk modulus @f$ \kappa @f$
  type_real mu;      ///< Shear modulus @f$ \mu @f$
  type_real nu;      ///< Symmetry breaking coupling modulus @f$ \nu @f$
  type_real j;       ///< Intertia density  @f$ j @f$
  type_real kappa_c; ///< Coupling bulk modulus @f$ \kappa_c @f$
  type_real mu_c;    ///< Coupling shear modulus @f$ \mu_c @f$
  type_real nu_c;    ///< Coupling symmetry breaking modulus @f$ \nu_c @f$
};

} // namespace medium
} // namespace specfem
