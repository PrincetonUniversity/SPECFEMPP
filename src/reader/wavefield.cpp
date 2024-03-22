#include "reader/wavefield.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "reader/reader.hpp"
#include "reader/wavefield.tpp"

// Explicit instantiation
template class specfem::reader::wavefield<
    specfem::IO::HDF5<specfem::IO::read> >;
