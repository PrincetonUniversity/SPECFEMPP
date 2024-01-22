#ifndef _ACOUSTIC_MATERIAL_HPP
#define _ACOUSTIC_MATERIAL_HPP

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "material.hpp"
#include "point/properties.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

namespace specfem {
namespace material {
/**
 * @brief Acoustic material class
 *
 * Defines the routines required to read and assign acoustic material properties
 * to specfem mesh.
 */
class acoustic_isotropic_material
    : public material<acoustic_isotropic_material> {
public:
  constexpr static specfem::enums::element::type type =
      specfem::enums::element::type::acoustic; ///< Type of element
  constexpr static specfem::enums::element::property_tag property =
      specfem::enums::element::property_tag::isotropic; ///< Property of element

  /**
   * @brief Construct a new acoustic material object
   *
   */
  acoustic_isotropic_material();

  /**
   * @brief Constructs a new acoustic material object
   * @param density Density of the material
   * @param cp Compressional wave speed
   * @param Qkappa Kappa attenuation factor
   * @param Qmu Mu attenuation factor
   * @param compaction_grad compaction gradient
   *
   */
  acoustic_isotropic_material(const type_real &density, const type_real &cp,
                              const type_real &Qkappa, const type_real &Qmu,
                              const type_real &compaction_grad);

  /**
   * @brief Get the type of the material
   *
   * @return specfem::enums::element::type The type of the material
   */
  specfem::enums::element::type get_type() const { return type; };

  specfem::point::properties<type, property> get_properties() const;

  /**
   * @brief Print material information to the console
   *
   * @return std::string String containing the material information
   */
  std::string print() const;

private:
  /**
   * @brief Acoustic material properties
   *
   */

  ///@{
  type_real density;
  type_real cp;
  type_real Qkappa;
  type_real Qmu;
  type_real compaction_grad;
  type_real lambdaplus2mu;
  type_real lambda;
  type_real kappa;
  type_real young;
  type_real poisson;
  ///@}
};

} // namespace material
} // namespace specfem

#endif
