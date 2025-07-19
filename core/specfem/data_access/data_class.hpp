#pragma once

namespace specfem::data_access {
enum DataClassType {
  index,
  assembly_index,
  mapped_index,
  properties,
  kernels,
  jacobian_matrix,
  field_derivatives,
  field,
  displacement,
  velocity,
  acceleration,
  mass_matrix,
  source,
  stress,
  stress_integrand,
  boundary
};
} // namespace specfem::data_access
