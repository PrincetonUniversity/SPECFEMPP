#ifndef _ACOUSTIC_MATERIAL_HPP
#define _ACOUSTIC_MATERIAL_HPP

#include "constants.hpp"
#include "material.hpp"
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
   * @brief Assign acoustic material values
   *
   * @param holder holder used to hold read values
   */
  void assign(utilities::input_holder &holder) override;
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);
  specfem::elements::type get_ispec_type() { return ispec_type; };
  std::string print() const override;

private:
  /**
   * @brief Acoustic material properties
   *
   */
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
  specfem::elements::type ispec_type = specfem::elements::acoustic;
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::material::acoustic_material &h);

} // namespace material
} // namespace specfem

#endif
