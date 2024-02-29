#ifndef READER_READER_HPP
#define READER_READER_HPP

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
  virtual void read() = 0;
};
} // namespace reader
} // namespace specfem

#endif
