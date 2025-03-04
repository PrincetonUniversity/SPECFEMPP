#pragma once

namespace specfem {
namespace IO {
namespace impl {
template <typename OutputLibrary, typename ContainerType>
void medium_writer(const std::string &output_folder,
                   const std::string &output_namespace,
                   const specfem::compute::assembly &assembly,
                   ContainerType &container);
} // namespace impl
} // namespace IO
} // namespace specfem
