#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
template <specfem::dimension::type DimensionTag> class assembly;
}

namespace specfem {
namespace io {
/**
 * @brief Base reader class
 *
 */
class reader {
public:
  /**
   * @brief Method to execute the read operation
   *
   * @param assembly Assembly object
   *
   */
  virtual void read(specfem::assembly::assembly<specfem::dimension::type::dim2>
                        &assembly) = 0;
};
} // namespace io
} // namespace specfem
