#include "IO/property/reader.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/property/reader.tpp"
#include "IO/reader.hpp"

// Explicit instantiation
template class specfem::IO::property_reader<
    specfem::IO::HDF5<specfem::IO::read> >;

template class specfem::IO::property_reader<
    specfem::IO::ASCII<specfem::IO::read> >;
