#pragma once

#include "io/property/writer.hpp"
#include "io/impl/medium_writer.hpp"

namespace specfem {
namespace io {

template <typename OutputLibrary>
kernel_writer<OutputLibrary>::kernel_writer(const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void kernel_writer<OutputLibrary>::write(specfem::assembly::assembly &assembly) {
  impl::write_container<OutputLibrary>(output_folder, "Kernels", assembly.mesh, assembly.element_types, assembly.kernels);
}

} // namespace io
} // namespace specfem
