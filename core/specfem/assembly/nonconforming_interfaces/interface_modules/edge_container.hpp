#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly::interface {

namespace module {

template <specfem::dimension::type DimensionTag> struct single_edge_container;

template <specfem::dimension::type DimensionTag> struct double_edge_container;
}; // namespace module
} // namespace specfem::assembly::interface

#include "dim2/edge_container.hpp"
