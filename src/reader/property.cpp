#include "reader/property.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "reader/property.tpp"
#include "reader/reader.hpp"

// Explicit instantiation
template class specfem::reader::property<specfem::IO::HDF5<specfem::IO::read> >;

template class specfem::reader::property<
    specfem::IO::ASCII<specfem::IO::read> >;
