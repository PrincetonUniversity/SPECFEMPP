#include "writer/property.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/property.tpp"

// Explicit instantiation

template class specfem::writer::property<
    specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::writer::property<
    specfem::IO::ASCII<specfem::IO::write> >;
