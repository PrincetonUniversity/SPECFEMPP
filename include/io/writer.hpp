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
   * @param assembly 2D Assembly object
   *
   */
  virtual void write(specfem::assembly::assembly<specfem::dimension::type::dim2>
                         &assembly) = 0;

  /**
   * @brief Method to execute the write operation
   *
   * @param assembly 3D Assembly object
   *
   */
  virtual void write(specfem::assembly::assembly<specfem::dimension::type::dim3>
                         &assembly) = 0;
};

} // namespace io
} // namespace specfem
