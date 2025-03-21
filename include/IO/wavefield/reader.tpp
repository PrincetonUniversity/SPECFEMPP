#pragma once

#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/wavefield/reader.hpp"

template <typename IOLibrary>
specfem::io::wavefield_reader<IOLibrary>::wavefield_reader(
    const std::string &output_folder)
    : output_folder(output_folder) {}

template <typename IOLibrary>
void specfem::io::wavefield_reader<IOLibrary>::read(
    specfem::compute::assembly &assembly) {
  auto &buffer = assembly.fields.buffer;
  auto &boundary_values = assembly.boundary_values;

  typename IOLibrary::File file(output_folder + "/ForwardWavefield");

  typename IOLibrary::Group elastic_sv = file.openGroup("/ElasticSV");

  const auto &elastic_sv_field =
      buffer.get_field<specfem::element::medium_tag::elastic_sv>();

  elastic_sv.openDataset("Displacement", elastic_sv_field.h_field).read();
  elastic_sv.openDataset("Velocity", elastic_sv_field.h_field_dot).read();
  elastic_sv.openDataset("Acceleration", elastic_sv_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group elastic_sh = file.openGroup("/ElasticSH");

  const auto &elastic_sh_field =
      buffer.get_field<specfem::element::medium_tag::elastic_sh>();

  elastic_sh.openDataset("Displacement", elastic_sh_field.h_field).read();
  elastic_sh.openDataset("Velocity", elastic_sh_field.h_field_dot).read();
  elastic_sh.openDataset("Acceleration", elastic_sh_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group acoustic = file.openGroup("/Acoustic");
  const auto &acoustic_field =
      buffer.get_field<specfem::element::medium_tag::acoustic>();

  acoustic.openDataset("Potential", acoustic_field.h_field).read();
  acoustic.openDataset("PotentialDot", acoustic_field.h_field_dot).read();
  acoustic.openDataset("PotentialDotDot", acoustic_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group boundary = file.openGroup("/Boundary");
  typename IOLibrary::Group stacey = boundary.openGroup("/Stacey");

  stacey
      .openDataset("IndexMapping",
                   boundary_values.stacey.h_property_index_mapping)
      .read();
  stacey
      .openDataset("ElasticAcceleration",
                   boundary_values.stacey.elastic.h_values)
      .read();
  stacey
      .openDataset("AcousticAcceleration",
                   boundary_values.stacey.acoustic.h_values)
      .read();

  buffer.copy_to_device();
  boundary_values.copy_to_device();
}
