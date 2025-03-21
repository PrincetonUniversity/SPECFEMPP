#include "IO/property/reader.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/property/reader.tpp"
#include "IO/reader.hpp"

// Explicit instantiation
template class specfem::io::property_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::ASCII<specfem::io::read> >;
