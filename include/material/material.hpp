#ifndef _MATERIAL_HPP
#define _MATERIAL_HPP

#include "enums.h"
#include "specfem_mpi.h"
#include "specfem_setup.hpp"
#include "utils.h"
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
   * @brief Virtual function to assign values read from database file to
   * material class members
   *
   * @param holder holder used to hold read values
   */
  virtual void assign(utilities::input_holder &holder){};
  virtual utilities::return_holder get_properties() {
    utilities::return_holder holder{};
    return holder;
  };
  virtual specfem::elements::type get_ispec_type() {
    return specfem::elements::elastic;
  };

  virtual std::string print() const { return ""; }
};

} // namespace material
} // namespace specfem

#endif
