#ifndef SPECFEM_WAVEFIELD_READER_TPP
#define SPECFEM_WAVEFIELD_READER_TPP

#include "IO/HDF5/HDF5.hpp"
#include "reader/wavefield.hpp"

template <typename IOLibrary>
specfem::reader::wavefield<IOLibrary>::wavefield(
    const std::string &output_folder,
    const specfem::compute::assembly &assembly)
    : output_folder(output_folder), buffer(assembly.fields.buffer) {}

template <typename IOLibrary>
void specfem::reader::wavefield<IOLibrary>::read() {

  typename IOLibrary::File file(output_folder + "/ForwardWavefield");

  typename IOLibrary::Group elastic = file.openGroup("/Elastic");

  elastic.openDataset("Displacement", buffer.elastic.field).read();
  elastic.openDataset("Velocity", buffer.elastic.field_dot).read();
  elastic.openDataset("Acceleration", buffer.elastic.field_dot_dot).read();

  typename IOLibrary::Group acoustic = file.openGroup("/Acoustic");

  acoustic.openDataset("Potential", buffer.acoustic.field).read();
  acoustic.openDataset("PotentialDot", buffer.acoustic.field_dot).read();
  acoustic.openDataset("PotentialDotDot", buffer.acoustic.field_dot_dot).read();
}

#endif /* SPECFEM_WAVEFIELD_READER_TPP */
