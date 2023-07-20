#ifndef _ACOUSTIC_MATERIAL_HPP
#define _ACOUSTIC_MATERIAL_HPP

#include "constants.hpp"
#include "material.hpp"
#include "specfem_enums.hpp"
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
class acoustic_material : public material {
public:
  /**
   * @brief Construct a new acoustic material object
   *
   */
  acoustic_material();
  /**
   * @brief Constructs a new acoustic material object
   * @param density Density of the material
   * @param cp Compressional wave speed
   * @param Qkappa Kappa attenuation factor
   * @param Qmu Mu attenuation factor
   * @param compaction_grad compaction gradient
   *
   */
  acoustic_material(const type_real &density, const type_real &cp,
                    const type_real &Qkappa, const type_real &Qmu,
                    const type_real &compaction_grad);
  /**
   * @brief ostream operator for acoustic material
   *
   * @param out Output stream
   * @param h Acoustic material object
   * @return std::ostream& Output stream
   */
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);
  /**
   * @brief Get the type of the material
   *
   * @return specfem::enums::element::type The type of the material
   */
  specfem::enums::element::type get_ispec_type() override {
    return ispec_type;
  };
  /**
   * @brief Get private elastic material properties
   *
   * @return utilities::return_holder holder used to return elastic material
   * properties
   */
  utilities::return_holder get_properties() override;
  /**
   * @brief Print material information to the console
   *
   * @return std::string String containing the material information
   */
  std::string print() const override;

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
  specfem::enums::element::type ispec_type =
      specfem::enums::element::acoustic; ///< Type or element ==
                                         ///< specfem::acoustic
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::material::acoustic_material &h);

} // namespace material
} // namespace specfem

#endif
