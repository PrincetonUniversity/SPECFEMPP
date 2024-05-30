#ifndef SPECFEM_WAVEFIELD_READER_TPP
#define SPECFEM_WAVEFIELD_READER_TPP

#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "reader/wavefield.hpp"

template <typename IOLibrary>
specfem::reader::wavefield<IOLibrary>::wavefield(
    const std::string &output_folder,
    const specfem::compute::assembly &assembly)
    : output_folder(output_folder), buffer(assembly.fields.buffer),
      boundary_values(assembly.boundary_values) {}

template <typename IOLibrary>
void specfem::reader::wavefield<IOLibrary>::read() {

  typename IOLibrary::File file(output_folder + "/ForwardWavefield");

  typename IOLibrary::Group elastic = file.openGroup("/Elastic");

  elastic.openDataset("Displacement", buffer.elastic.h_field).read();
  elastic.openDataset("Velocity", buffer.elastic.h_field_dot).read();
  elastic.openDataset("Acceleration", buffer.elastic.h_field_dot_dot).read();

  typename IOLibrary::Group acoustic = file.openGroup("/Acoustic");

  acoustic.openDataset("Potential", buffer.acoustic.h_field).read();
  acoustic.openDataset("PotentialDot", buffer.acoustic.h_field_dot).read();
  acoustic.openDataset("PotentialDotDot", buffer.acoustic.h_field_dot_dot)
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

#endif /* SPECFEM_WAVEFIELD_READER_TPP */
