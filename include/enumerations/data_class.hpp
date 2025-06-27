#pragma once

namespace specfem::data_class {
enum type {
  index,
  mapped_index,
  properties,
  kernels,
  jacobian_matrix,
  field_derivatives,
  field,
  source,
  stress,
  stress_integrand,
  boundary
};
} // namespace specfem::data_class
