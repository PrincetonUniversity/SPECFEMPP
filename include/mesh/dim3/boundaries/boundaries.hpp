#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

template <> struct boundaries<specfem::dimension::type::dim3> {};

} // namespace mesh
} // namespace specfem
