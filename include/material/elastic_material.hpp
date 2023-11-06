#ifndef _ELASTIC_MATERIAL_HPP
#define _ELASTIC_MATERIAL_HPP

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
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
class elastic_material : public material {
public:
  /**
   * @brief Construct a new elastic material object
   *
   */
  elastic_material();
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
  elastic_material(const type_real &density, const type_real &cs,
                   const type_real &cp, const type_real &Qkappa,
                   const type_real &Qmu, const type_real &compaction_grad);
  /**
   * @brief User output
   * Prints the read material values and additional information on
   * console/output file
   *
   * @param out Empty output stream
   * @param h elastic material holder to pass read values
   * @return std::ostream& Output stream to be displayed
   */
  friend std::ostream &operator<<(std::ostream &out, const elastic_material &h);
  /**
   * @brief Get private elastic material properties
   *
   * @return utilities::return_holder holder used to return elastic material
   * properties
   */
  utilities::return_holder get_properties() override;
  specfem::enums::element::type get_ispec_type() override {
    return ispec_type;
  };
  /**
   * @brief Print material information to the console
   *
   * @return std::string String containing the material information
   */
  std::string print() const override;

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
  specfem::enums::element::type ispec_type =
      specfem::enums::element::type::elastic; ///< Type or element ==
                                              ///< specfem::elastic
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::material::elastic_material &h);

} // namespace material
} // namespace specfem

#endif
