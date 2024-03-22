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

  fields() = default;

  fields(const specfem::compute::mesh &mesh,
         const specfem::compute::properties &properties,
         const specfem::simulation::type simulation);

  template <specfem::wavefield::type fieldtype>
  KOKKOS_INLINE_FUNCTION specfem::compute::simulation_field<fieldtype>
  get_simulation_field() const {
    if constexpr (fieldtype == specfem::wavefield::type::forward) {
      return forward;
    } else if constexpr (fieldtype == specfem::wavefield::type::adjoint) {
      return adjoint;
    } else if constexpr (fieldtype == specfem::wavefield::type::backward) {
      return backward;
    } else if constexpr (fieldtype == specfem::wavefield::type::buffer) {
      return buffer;
    } else {
      static_assert("field type not supported");
    }
  }

  template <specfem::sync::kind sync> void sync_fields() {
    forward.sync_fields<sync>();
    adjoint.sync_fields<sync>();
    backward.sync_fields<sync>();
    buffer.sync_fields<sync>();
  }

  specfem::compute::simulation_field<specfem::wavefield::type::buffer> buffer;
  specfem::compute::simulation_field<specfem::wavefield::type::forward> forward;
  specfem::compute::simulation_field<specfem::wavefield::type::adjoint> adjoint;
  specfem::compute::simulation_field<specfem::wavefield::type::backward>
      backward;
};

} // namespace compute
} // namespace specfem

#endif /* _COMPUTE_FIELDS_FIELDS_HPP_ */
