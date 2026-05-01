#include "specfem/io/property/writer.hpp"
#include "specfem/io/impl/medium_writer.tpp"
#include "specfem/io/property/writer.tpp"
#include "specfem/io_backends/ADIOS2/ADIOS2.hpp"
#include "specfem/io_backends/ASCII/ASCII.hpp"
#include "specfem/io_backends/HDF5/HDF5.hpp"
#include "specfem/io_backends/NPY/NPY.hpp"
#include "specfem/io_backends/NPZ/NPZ.hpp"

// Explicit instantiation

template class specfem::io::property_writer<
    specfem::io_backends::HDF5<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io_backends::ADIOS2<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io_backends::ASCII<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io_backends::NPY<specfem::io::write> >;

template class specfem::io::property_writer<
    specfem::io_backends::NPZ<specfem::io::write> >;
