#ifndef _SPECFEM_BOUNDARY_BOUNDARY_HPP
#define _SPECFEM_BOUNDARY_BOUNDARY_HPP

#include "enumerations/boundary.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace boundary {
template <specfem::wavefield::type WavefieldType,
          specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag,
          typename quadrature_points_type>
class boundary;
} // namespace boundary
} // namespace specfem

#endif // _SPECFEM_BOUNDARY_BOUNDARY_HPP
