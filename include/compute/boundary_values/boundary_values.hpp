#ifndef _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_HPP
#define _COMPUTE_BOUNDARIES_VALUES_BOUNDARY_VALUES_HPP

#include "boundary_values_container.hpp"
#include "compute/boundaries/boundaries.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/properties/properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "impl/boundary_medium_container.hpp"

namespace specfem {
namespace compute {
class boundary_values {
public:
  boundary_values() = default;

  specfem::compute::boundary_value_container<
      specfem::dimension::type::dim2, specfem::element::boundary_tag::stacey>
      stacey;

  boundary_values(const int nstep, const specfem::compute::mesh mesh,
                  const specfem::compute::properties properties,
                  const specfem::compute::boundaries boundaries);
};
} // namespace compute
} // namespace specfem

#endif
