#include "io/property/reader.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/Npy/Npy.hpp"
#include "io/property/reader.tpp"
#include "io/reader.hpp"

// Explicit instantiation
template class specfem::io::property_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::ASCII<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::Npy<specfem::io::read> >;
