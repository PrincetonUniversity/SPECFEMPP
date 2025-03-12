#pragma once

#include "IO/wavefield/writer.hpp"
#include "compute/interface.hpp"
#include "enumerations/interface.hpp"

template <typename OutputLibrary>
specfem::IO::wavefield_writer<OutputLibrary>::wavefield_writer(
    const std::string output_folder)
    : output_folder(output_folder) {}

template <typename OutputLibrary>
void specfem::IO::wavefield_writer<OutputLibrary>::write(
    specfem::compute::assembly &assembly) {
  auto &forward = assembly.fields.forward;
  auto &boundary_values = assembly.boundary_values;

  forward.copy_to_host();
  boundary_values.copy_to_host();

  typename OutputLibrary::File file(output_folder + "/ForwardWavefield");

  typename OutputLibrary::Group elastic_sv = file.createGroup("/ElasticSV");
  const auto &elastic_sv_field =
      forward.get_field<specfem::element::medium_tag::elastic_sv>();
  typename OutputLibrary::Group elastic_sh = file.createGroup("/ElasticSH");
  const auto &elastic_sh_field =
      forward.get_field<specfem::element::medium_tag::elastic_sh>();
  typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");
  const auto &acoustic_field =
      forward.get_field<specfem::element::medium_tag::acoustic>();
  typename OutputLibrary::Group boundary = file.createGroup("/Boundary");
  typename OutputLibrary::Group stacey = boundary.createGroup("/Stacey");

  elastic_sv.createDataset("Displacement", elastic_sv_field.h_field).write();
  elastic_sv.createDataset("Velocity", elastic_sv_field.h_field_dot).write();
  elastic_sv.createDataset("Acceleration", elastic_sv_field.h_field_dot_dot)
      .write();

  elastic_sh.createDataset("Displacement", elastic_sh_field.h_field).write();
  elastic_sh.createDataset("Velocity", elastic_sh_field.h_field_dot).write();
  elastic_sh.createDataset("Acceleration", elastic_sh_field.h_field_dot_dot)
      .write();

  acoustic.createDataset("Potential", acoustic_field.h_field).write();
  acoustic.createDataset("PotentialDot", acoustic_field.h_field_dot).write();
  acoustic.createDataset("PotentialDotDot", acoustic_field.h_field_dot_dot).write();

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
