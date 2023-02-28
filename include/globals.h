#ifndef GLOBALS_H
#define GLOBALS_H

#include "../include/enums.h"

namespace specfem {
namespace globals {
const specfem::wave::type simulation_wave = specfem::wave::p_sv;
static specfem::kokkos::DeviceView1d<specfem::seismogram::type>
    seismogram_types;
static specfem::kokkos::HostMirror1d<specfem::seismogram::type>
    h_seismogram_types;
} // namespace globals
} // namespace specfem

#endif
