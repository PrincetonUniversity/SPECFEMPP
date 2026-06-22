#pragma once

#include "specfem/io/property/writer.hpp"
#include "specfem/io/impl/medium_writer.hpp"
#include "writer.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/mpi.hpp"
#include <boost/filesystem.hpp>

namespace specfem {
namespace io {

template <typename OutputLibrary>
property_writer<OutputLibrary>::property_writer(const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void property_writer<OutputLibrary>::write(specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &assembly) {
  // Build rank-specific output path following the same convention as mesh files:
  //   serial:   {output_folder}/Properties/
  //   parallel: {output_folder}/Properties/proc_N/
  // format_proc_filename("foo.ext") -> "foo/proc_N.ext" (nproc>1) or "foo.ext" (nproc==1)
  const std::string formatted =
      specfem::MPI::format_proc_filename(output_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  // Ensure intermediate directories exist (needed for MPI case where
  // base_folder = output_folder/Properties/ which may not exist yet)
  boost::filesystem::create_directories(base_folder);

  impl::write_container<OutputLibrary>(base_folder, ns,
      assembly.mesh, assembly.element_types, assembly.properties);
}

template <typename OutputLibrary>
void property_writer<OutputLibrary>::write(specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &assembly) {
  // Build rank-specific output path following the same convention as mesh files:
  //   serial:   {output_folder}/Properties/
  //   parallel: {output_folder}/Properties/proc_N/
  const std::string formatted =
      specfem::MPI::format_proc_filename(output_folder + "/Properties");
  const boost::filesystem::path formatted_path(formatted);
  const std::string base_folder = formatted_path.parent_path().string();
  const std::string ns = formatted_path.stem().string();

  boost::filesystem::create_directories(base_folder);

  impl::write_container<OutputLibrary>(base_folder, ns,
      assembly.mesh, assembly.element_types, assembly.properties);
}

} // namespace io
} // namespace specfem
