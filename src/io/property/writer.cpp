#include "io/property/writer.hpp"
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/NPY/NPY.hpp"
#include "io/NPZ/NPZ.hpp"
#include "io/impl/medium_writer.tpp"
#include "io/property/writer.tpp"

// Explicit instantiation

template class specfem::io::property_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::ADIOS2<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::ASCII<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::NPY<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io::NPZ<specfem::io::write> >;
