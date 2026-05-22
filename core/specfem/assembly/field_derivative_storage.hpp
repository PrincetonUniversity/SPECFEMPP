#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag>
struct FieldDerivativeStorage;

} // namespace specfem::assembly

#include "specfem/assembly/field_derivative_storage/dim2/field_derivative_storage.hpp"
#include "specfem/assembly/field_derivative_storage/dim3/field_derivative_storage.hpp"
