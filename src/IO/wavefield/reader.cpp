#include "IO/wavefield/reader.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/reader.hpp"
#include "IO/wavefield/reader.tpp"

// Explicit instantiation
template class specfem::IO::wavefield_reader<
    specfem::IO::HDF5<specfem::IO::read> >;

template class specfem::IO::wavefield_reader<
    specfem::IO::ASCII<specfem::IO::read> >;
