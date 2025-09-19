#include "io/property/reader.hpp"
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/NPY/NPY.hpp"
#include "io/NPZ/NPZ.hpp"
#include "io/property/reader.tpp"
#include "io/reader.hpp"

// Explicit instantiation
template class specfem::io::property_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::ADIOS2<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::ASCII<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::NPY<specfem::io::read> >;

template class specfem::io::property_reader<
    specfem::io::NPZ<specfem::io::read> >;
