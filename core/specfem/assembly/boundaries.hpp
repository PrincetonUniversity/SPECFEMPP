#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::boundaries_impl {

template <specfem::dimension::type DimensionTag> struct acoustic_free_surface;

template <specfem::dimension::type DimensionTag> struct stacey;

} // namespace specfem::assembly::boundaries_impl

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag> class boundaries;

}

#include "boundaries/boundaries.hpp"
#include "boundaries/impl/acoustic_free_surface.hpp"
#include "boundaries/impl/stacey.hpp"
