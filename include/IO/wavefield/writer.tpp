#pragma once

#include "compute/interface.hpp"
#include "enumerations/interface.hpp"
#include "IO/wavefield/writer.hpp"

template <typename OutputLibrary>
specfem::IO::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::IO::wavefield_writer<OutputLibrary>::write(specfem::compute::assembly &assembly) {
  auto &forward = assembly.fields.forward;
  auto &boundary_values = assembly.boundary_values;

  forward.copy_to_host();
  boundary_values.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/ForwardWavefield");

  typename OutputLibrary::Group elastic = file.createGroup("/Elastic");
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");
  typename OutputLibrary::Group boundary = file.createGroup("/Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("/Stacey");

  elastic.createDataset("Displacement", forward.elastic.h_field).write();
  elastic.createDataset("Velocity", forward.elastic.h_field_dot).write();
  elastic.createDataset("Acceleration", forward.elastic.h_field_dot_dot).write();

  acoustic.createDataset("Potential", forward.acoustic.h_field).write();
  acoustic.createDataset("PotentialDot", forward.acoustic.h_field_dot).write();
  acoustic.createDataset("PotentialDotDot", forward.acoustic.h_field_dot_dot)
      .write();

  stacey
      .createDataset("IndexMapping",
                     boundary_values.stacey.h_property_index_mapping)
      .write();
  stacey
      .createDataset("ElasticAcceleration",
                     boundary_values.stacey.elastic.h_values)
      .write();
  stacey
      .createDataset("AcousticAcceleration",
                     boundary_values.stacey.acoustic.h_values)
      .write();

  std::cout << "Wavefield written to " << output_folder + "/ForwardWavefield"
            << std::endl;
}
