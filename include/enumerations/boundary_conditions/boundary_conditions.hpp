#ifndef _SPECFEM_BOUNDARY_BOUNDARY_HPP
#define _SPECFEM_BOUNDARY_BOUNDARY_HPP

#include "enumerations/boundary.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace boundary {
template <specfem::dimension::type dimension,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property,
          specfem::element::boundary_tag BoundaryType,
          typename quadrature_points_type>
class boundary;
} // namespace boundary
} // namespace specfem

#endif // _SPECFEM_BOUNDARY_BOUNDARY_HPP
