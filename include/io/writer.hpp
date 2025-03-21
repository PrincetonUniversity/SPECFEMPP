#pragma once

namespace specfem {
namespace compute {
class assembly;
}
} // namespace specfem

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
  virtual void write(specfem::compute::assembly &assembly) {};
};

} // namespace io
} // namespace specfem
