#pragma once

#include "IO/property/writer.hpp"
#include "IO/impl/medium_writer.hpp"

namespace specfem {
namespace IO {

template <typename OutputLibrary>
property_writer<OutputLibrary>::property_writer(const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void property_writer<OutputLibrary>::write(specfem::compute::assembly &assembly) {
  impl::write_container<OutputLibrary>(output_folder, "Properties", assembly.mesh, assembly.element_types, assembly.properties);
}

} // namespace IO
} // namespace specfem
