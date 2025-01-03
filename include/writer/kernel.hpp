#ifndef _SPECFEM_WRITER_KERNEL_HPP
#define _SPECFEM_WRITER_KERNEL_HPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {
/**
 * @brief Writer to output misfit kernel data to disk
 *
 * @tparam OutputLibrary Library to use for output (HDF5, ASCII, etc.)
 */
template <typename OutputLibrary> class kernel : public writer {
public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a writer object
   *
   * @param assembly SPECFEM++ assembly
   * @param output_folder Path to output location (will be an .h5 file if using
   * HDF5, and a folder if using ASCII)
   */
  kernel(const std::string output_folder);

  /**
   * @brief write the kernel data to disk
   *
   */
  void write(specfem::compute::assembly &assembly) override;

private:
  std::string output_folder; ///< Path to output folder
};
} // namespace writer
} // namespace specfem

#endif /* _SPECFEM_WRITER_KERNEL_HPP */
