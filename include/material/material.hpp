#ifndef _MATERIAL_HPP
#define _MATERIAL_HPP

#include "constants.hpp"
#include "enumerations/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

namespace specfem {
namespace material {

/**
 * @brief Base material class
 *
 */
class material {
public:
  /**
   * @brief Construct a new material object
   *
   */
  material(){};
  /**
   * @brief Get the properties of the material
   *
   * @return utilities::return_holder Struct containing the properties of the
   * material
   */
  virtual utilities::return_holder get_properties() {
    utilities::return_holder holder{};
    return holder;
  };
  /**
   * @brief Get the type of the material
   *
   * @return specfem::enums::element::type The type of the material
   */
  virtual specfem::enums::element::type get_ispec_type() {
    return specfem::enums::element::elastic;
  };

  /**
   * @brief Print material information to the console
   *
   * @return std::string String containing the material information
   */
  virtual std::string print() const { return ""; }
};

} // namespace material
} // namespace specfem

#endif
