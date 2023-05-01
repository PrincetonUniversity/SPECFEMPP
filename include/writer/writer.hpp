#ifndef _WRITER_HPP
#define _WRITER_HPP

namespace specfem {
namespace writer {
/**
 * @brief Base writer class
 *
 */
class writer {
public:
  /**
   * @brief Method to execute the write operation
   *
   */
  virtual void write(){};
};

} // namespace writer
} // namespace specfem

#endif
