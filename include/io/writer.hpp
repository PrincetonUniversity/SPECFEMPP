#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {
template <specfem::dimension::type DimensionTag> class assembly;
}

namespace specfem {
namespace io {
/**
 * @brief Base writer class
 *
 */
class writer {
public:
  /**
   * @brief Method to execute the write operation
   *
   * @param assembly Assembly object
   *
   */
  virtual void write(specfem::assembly::assembly<specfem::dimension::type::dim2>
                         &assembly) = 0;
};

} // namespace io
} // namespace specfem
