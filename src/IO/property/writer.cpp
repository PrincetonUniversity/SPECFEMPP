#include "IO/property/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/impl/medium_writer.tpp"
#include "IO/property/writer.tpp"

// Explicit instantiation

template class specfem::io::property_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::ASCII<specfem::io::write> >;
