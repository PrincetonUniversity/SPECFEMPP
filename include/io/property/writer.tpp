#pragma once

#include "io/property/writer.hpp"
#include "io/impl/medium_writer.hpp"
#include "writer.hpp"
#include "specfem/assembly.hpp"
#include "enumerations/interface.hpp"

namespace specfem {
namespace io {

template <typename OutputLibrary>
property_writer<OutputLibrary>::property_writer(const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void property_writer<OutputLibrary>::write(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  impl::write_container<OutputLibrary>(output_folder, "Properties", assembly.mesh, assembly.element_types, assembly.properties);
}

} // namespace io
} // namespace specfem
