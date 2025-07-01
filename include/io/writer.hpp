#pragma once

namespace specfem::assembly {
class assembly;
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
  virtual void write(specfem::assembly::assembly &assembly) = 0;
};

} // namespace io
} // namespace specfem
