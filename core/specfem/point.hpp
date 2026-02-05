#pragma once

#include "point/acceleration.hpp"
#include "point/assembly_index.hpp"
#include "point/boundary.hpp"
#include "point/conforming_interface.hpp"
#include "point/displacement.hpp"
#include "point/edge_index.hpp"
#include "point/face_index.hpp"
#include "point/field_derivatives.hpp"
#include "point/global_coordinates.hpp"
#include "point/index.hpp"
#include "point/interface_index.hpp"
#include "point/jacobian_matrix.hpp"
#include "point/kernels.hpp"
#include "point/local_coordinates.hpp"
#include "point/mapped_index.hpp"
#include "point/mass_inverse.hpp"
#include "point/nonconforming_interface.hpp"
#include "point/properties.hpp"
#include "point/source.hpp"
#include "point/stress.hpp"
#include "point/stress_integrand.hpp"
#include "point/velocity.hpp"
#include "point/weights.hpp"

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
