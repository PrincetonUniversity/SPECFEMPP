#ifndef _ELASTIC_MATERIAL_HPP
#define _ELASTIC_MATERIAL_HPP

#include "enums.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utils.h"
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
   * @brief Assign elastic material values
   *
   * @param holder holder used to hold read values
   */
  void assign(utilities::input_holder &holder) override;
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
  specfem::elements::type get_ispec_type() { return ispec_type; };

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
  specfem::elements::type ispec_type =
      specfem::elements::elastic; ///< Type or element == specfem::elastic
};

std::ostream &operator<<(std::ostream &out,
                         const specfem::material::elastic_material &h);

} // namespace material
} // namespace specfem

#endif
