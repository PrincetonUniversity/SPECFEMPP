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
class acoustic_material : public material {
public:
  /**
   * @brief Construct a new acoustic material object
   *
   */
  acoustic_material();
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
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
  specfem::enums::element::type ispec_type = specfem::enums::element::acoustic;
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::material::acoustic_material &h);

} // namespace material
} // namespace specfem

#endif
