#ifndef SHAPE_FUNCTIONS_H
#define SHAPE_FUNCTIONS_H

#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

using HostView1d = specfem::HostView1d<type_real>;
using HostView2d = specfem::HostView2d<type_real>;

namespace shape_functions {

HostView1d define_shape_functions(const double xi, const double gamma,
                                  const int ngod);
HostView2d define_shape_functions_derivatives(const double xi,
                                              const double gamma,
                                              const int ngod);
} // namespace shape_functions

#endif // SHAPE_FUNCTIONS_H
