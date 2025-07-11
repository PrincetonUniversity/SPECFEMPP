#pragma once

namespace specfem {
namespace io {
namespace impl {
template <typename OutputLibrary, typename ContainerType>
void write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    ContainerType &container);
} // namespace impl
} // namespace io
} // namespace specfem
