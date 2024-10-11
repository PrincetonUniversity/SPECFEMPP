#ifndef _SPECFEM_WRITER_KERNEL_HPP
#define _SPECFEM_WRITER_KERNEL_HPP

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "writer/writer.hpp"

namespace specfem {
namespace writer {
template <typename OutputLibrary> class kernel : public writer {
public:
  kernel(const specfem::compute::assembly &assembly,
         const std::string output_folder);

  void write() override;

private:
  std::string output_folder; ///< Path to output folder
  specfem::compute::mesh mesh;
  specfem::compute::kernels kernels;
};
} // namespace writer
} // namespace specfem

#endif /* _SPECFEM_WRITER_KERNEL_HPP */
