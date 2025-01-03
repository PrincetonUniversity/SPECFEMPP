#pragma once

namespace specfem {
namespace compute {
class assembly;
}
} // namespace specfem

namespace specfem {
namespace reader {
/**
 * @brief Base reader class
 *
 */
class reader {
public:
  /**
   * @brief Method to execute the read operation
   *
   */
  virtual void read(specfem::compute::assembly &assembly) = 0;
};
} // namespace reader
} // namespace specfem
