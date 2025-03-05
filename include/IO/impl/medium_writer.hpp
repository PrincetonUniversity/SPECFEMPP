#pragma once

namespace specfem {
namespace IO {
namespace impl {
template <typename OutputLibrary, typename ContainerType>
void write_container(const std::string &output_folder,
                     const std::string &output_namespace,
                     const specfem::compute::mesh &mesh,
                     const specfem::compute::element_types &element_types,
                     ContainerType &container);
} // namespace impl
} // namespace IO
} // namespace specfem
