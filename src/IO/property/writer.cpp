#include "IO/property/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/impl/medium_writer.tpp"
#include "IO/property/writer.tpp"

// Explicit instantiation

template class specfem::IO::property_writer<
    specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::IO::property_writer<
    specfem::IO::ASCII<specfem::IO::write> >;
