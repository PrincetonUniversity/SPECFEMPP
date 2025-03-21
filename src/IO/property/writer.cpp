#include "io/property/writer.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/impl/medium_writer.tpp"
#include "io/property/writer.tpp"

// Explicit instantiation

template class specfem::io::property_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::ASCII<specfem::io::write> >;
