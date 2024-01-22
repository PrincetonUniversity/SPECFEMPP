#ifndef _ELASTIC_MATERIAL_HPP
#define _ELASTIC_MATERIAL_HPP

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "point/properties.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

namespace specfem {
namespace material {
/**
 * @brief Elastic material class
 *
 * Defines the routines required to read and assign elastic material properties
 * to specfem mesh.
 */
class elastic_isotropic_material : public material<elastic_isotropic_material> {
public:
  constexpr static specfem::enums::element::type type =
      specfem::enums::element::type::elastic; ///< Type of element
  constexpr static specfem::enums::element::property_tag property =
      specfem::enums::element::property_tag::isotropic; ///< Property of element

  /**
   * @brief Construct a new elastic material object
   *
   */
  elastic_isotropic_material();

  /**
   * @brief Construct a new elastic material object
   *
   * @param density Density of the material
   * @param cs Transverse wave speed
   * @param cp Compressional wave speed
   * @param Qkappa Kappa attenuation factor
   * @param Qmu Mu attenuation factor
   * @param compaction_grad compaction gradient
   */
  elastic_isotropic_material(const type_real &density, const type_real &cs,
                             const type_real &cp, const type_real &Qkappa,
                             const type_real &Qmu,
                             const type_real &compaction_grad);

  std::string print() const;

  /**
   * @brief get material properties
   *
   * @return specfem::point::property<type, property>
   */

  specfem::point::properties<type, property> get_properties() const;

  constexpr specfem::enums::element::type get_type() const { return type; };

private:
  /**
   * @name Elastic material properties
   *
   */
  ///@{
  type_real density;
  type_real cs;
  type_real cp;
  type_real Qkappa;
  type_real Qmu;
  type_real compaction_grad;
  type_real lambdaplus2mu;
  type_real mu;
  type_real lambda;
  type_real kappa;
  type_real young;
  type_real poisson;
  ///@}
};

} // namespace material
} // namespace specfem

#endif
