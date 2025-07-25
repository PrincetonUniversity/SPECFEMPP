#pragma once

#include "enumerations/interface.hpp"

namespace specfem::sources {

enum class source_type {
  vector_source, ///< Vector source
  tensor_source, ///< Tensor source
};

}

namespace specfem::sources {

template <specfem::dimension::type DimensionTag> class tensor_source;

template <specfem::dimension::type DimensionTag> class vector_source;

template <specfem::dimension::type DimensionTag> struct moment_tensor;

template <specfem::dimension::type DimensionTag> class force;

template <specfem::dimension::type DimensionTag> class cosserat_force;

template <specfem::dimension::type DimensionTag> class adjoint_source;

template <specfem::dimension::type DimensionTag> class external;

} // namespace specfem::sources

#include "source/source.hpp"
#include "source/tensor_source.hpp"
#include "source/vector_source.hpp"

// dim2 specializations
#include "source/dim2/source.tpp"
#include "source/dim2/tensor_source/moment_tensor_source.hpp"
#include "source/dim2/vector_source/adjoint_source.hpp"
#include "source/dim2/vector_source/cosserat_force_source.hpp"
#include "source/dim2/vector_source/external.hpp"
#include "source/dim2/vector_source/force_source.hpp"

// dim3 specializations
#include "source/dim3/source.tpp"
#include "source/dim3/tensor_source/moment_tensor_source.hpp"
#include "source/dim3/vector_source/force_source.hpp"
