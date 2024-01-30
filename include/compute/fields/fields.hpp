#ifndef _COMPUTE_FIELDS_FIELDS_HPP_
#define _COMPUTE_FIELDS_FIELDS_HPP_

#include "compute/compute_mesh.hpp"
#include "compute/properties/interface.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/specfem_enums.hpp"
#include "simulation_field.hpp"
#include "simulation_field.tpp"

namespace specfem {
namespace compute {
struct fields {

  using forward_type = specfem::enums::simulation::forward;

  fields() = default;

  fields(const specfem::compute::mesh &mesh,
         const specfem::compute::properties &properties);

  template <typename simulation>
  KOKKOS_INLINE_FUNCTION specfem::compute::simulation_field<simulation>
  get_simulation_field() {
    if constexpr (std::is_same_v<simulation, forward_type>) {
      return forward;
    }
  }

  specfem::compute::simulation_field<forward_type> forward;
};
} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_FIELDS_HPP_ */
