#pragma once

namespace specfem::data_class {
enum type {
  index,
  properties,
  kernels,
  partial_derivatives,
  field_derivatives,
  field,
  source,
  stress,
  stress_integrand,
  boundary
};
} // namespace specfem::data_class
