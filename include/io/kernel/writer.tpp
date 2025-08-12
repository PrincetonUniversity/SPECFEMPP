#pragma once

#include "io/impl/medium_writer.hpp"
#include "io/property/writer.hpp"
#include "writer.hpp"
#include "specfem/assembly.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace io {

template <typename OutputLibrary>
kernel_writer<OutputLibrary>::kernel_writer(const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void kernel_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  impl::write_container<OutputLibrary>(output_folder, "Kernels", assembly.mesh,
                                       assembly.element_types,
                                       assembly.kernels);
}

} // namespace io
} // namespace specfem
