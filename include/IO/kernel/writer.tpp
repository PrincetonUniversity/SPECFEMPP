#pragma once

#include "IO/property/writer.hpp"
#include "IO/impl/medium_writer.hpp"

namespace specfem {
namespace IO {

template <typename OutputLibrary>
kernel_writer<OutputLibrary>::kernel_writer(const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void kernel_writer<OutputLibrary>::write(specfem::compute::assembly &assembly) {
  impl::write_container<OutputLibrary>(output_folder, "Kernels", assembly.mesh, assembly.element_types, assembly.kernels);
}

} // namespace IO
} // namespace specfem
