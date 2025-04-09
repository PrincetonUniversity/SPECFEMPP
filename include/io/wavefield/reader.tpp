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

  typename IOLibrary::Group elastic_psv = file.openGroup("/ElasticPSV");

  const auto &elastic_psv_field =
      buffer.get_field<specfem::element::medium_tag::elastic_psv>();

  elastic_psv.openDataset("Displacement", elastic_psv_field.h_field).read();
  elastic_psv.openDataset("Velocity", elastic_psv_field.h_field_dot).read();
  elastic_psv.openDataset("Acceleration", elastic_psv_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group elastic_sh = file.openGroup("/ElasticSH");

  const auto &elastic_sh_field =
      buffer.get_field<specfem::element::medium_tag::elastic_sh>();

  elastic_sh.openDataset("Displacement", elastic_sh_field.h_field).read();
  elastic_sh.openDataset("Velocity", elastic_sh_field.h_field_dot).read();
  elastic_sh.openDataset("Acceleration", elastic_sh_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group elastic_psv_t = file.openGroup("/ElasticPSVT");
  const auto &elastic_psv_t_field =
      buffer.get_field<specfem::element::medium_tag::elastic_psv_t>();

  elastic_psv_t.openDataset("Displacement", elastic_psv_t_field.h_field).read();
  elastic_psv_t.openDataset("Velocity", elastic_psv_t_field.h_field_dot).read();
  elastic_psv_t.openDataset("Acceleration", elastic_psv_t_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group acoustic = file.openGroup("/Acoustic");
  const auto &acoustic_field =
      buffer.get_field<specfem::element::medium_tag::acoustic>();

  acoustic.openDataset("Potential", acoustic_field.h_field).read();
  acoustic.openDataset("PotentialDot", acoustic_field.h_field_dot).read();
  acoustic.openDataset("PotentialDotDot", acoustic_field.h_field_dot_dot)
      .read();

  typename IOLibrary::Group poroelastic = file.openGroup("/Poroelastic");

  const auto &poroelastic_field =
      buffer.get_field<specfem::element::medium_tag::poroelastic>();

  poroelastic.openDataset("Displacement", poroelastic_field.h_field).read();
  poroelastic.openDataset("Velocity", poroelastic_field.h_field_dot).read();
  poroelastic.openDataset("Acceleration", poroelastic_field.h_field_dot_dot)
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
  stacey
      .openDataset("PoroelasticAcceleration",
                   boundary_values.stacey.poroelastic.h_values)
      .read();

  buffer.copy_to_device();
  boundary_values.copy_to_device();
}
