#pragma once

#include "point/assembly_index.hpp"
#include "point/boundary.hpp"
#include "point/coordinates.hpp"
#include "point/field_derivatives.hpp"
#include "point/index.hpp"
#include "point/jacobian_matrix.hpp"
#include "point/kernels.hpp"
#include "point/mapped_index.hpp"
#include "point/properties.hpp"
#include "point/source.hpp"
#include "point/stress.hpp"
#include "point/stress_integrand.hpp"

/**
 * @namespace specfem::point
 *
 * @brief Namespace for structures that hold data at a single quadrature point
 *
 * This namespace contains various structures and functions that are used to
 * hold and manipulate data at a single quadrature point in the context of
 * finite element methods. It includes data structures for properties, fields,
 * kernels, and other related entities that are essential for numerical
 * simulations in geophysics and other fields.
 *
 */
namespace specfem::point {}
