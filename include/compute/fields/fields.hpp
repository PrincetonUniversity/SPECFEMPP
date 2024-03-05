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

  constexpr static auto forward_type = specfem::simulation::type::forward;
  constexpr static auto adjoint_type = specfem::simulation::type::adjoint;

  fields() = default;

  fields(const specfem::compute::mesh &mesh,
         const specfem::compute::properties &properties,
         const specfem::simulation::type simulation);

  template <specfem::simulation::type simulation>
  KOKKOS_INLINE_FUNCTION specfem::compute::simulation_field<simulation>
  get_simulation_field() {
    if constexpr (std::is_same_v<simulation, forward_type>) {
      return forward;
    } else if constexpr (std::is_same_v<simulation, adjoint_type>) {
      return adjoint;
    }
  }

  template <specfem::sync::kind sync> void sync_fields() {
    forward.sync_fields<sync>();
    adjoint.sync_fields<sync>();
  }

  specfem::compute::simulation_field<forward_type> forward;
  specfem::compute::simulation_field<adjoint_type> adjoint;
};

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_FIELDS_HPP_ */
