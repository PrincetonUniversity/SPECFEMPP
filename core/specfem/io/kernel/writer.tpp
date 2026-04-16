#pragma once

#include "specfem/io/impl/medium_writer.hpp"
#include "specfem/io/property/writer.hpp"
#include "writer.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mpi.hpp"
#include <boost/filesystem.hpp>

namespace specfem {
namespace io {

template <typename OutputLibrary>
kernel_writer<OutputLibrary>::kernel_writer(const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void kernel_writer<OutputLibrary>::write(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  // Build rank-specific output path following the same convention as mesh files:
  //   serial:   {output_folder}/Kernels/
  //   parallel: {output_folder}/Kernels/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(output_folder + "/Kernels");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  boost::filesystem::create_directories(base_folder);

  impl::write_container<OutputLibrary>(base_folder, ns, assembly.mesh,
                                       assembly.element_types,
                                       assembly.kernels);
}

} // namespace io
} // namespace specfem
