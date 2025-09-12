#pragma once

#include "enumerations/interface.hpp"

/**
 * @namespace specfem::sources
 * @brief Namespace for structures that hold data related to sources
 *
 * This namespace contains source classes in a hierarchical structure. The base
 * class is @ref specfem::sources::source, with second-level
 * @ref specfem::sources::vector_source and @ref
 * specfem::sources::tensor_source, abstractions followed by their specific
 * implementations.
 *
 * <b>2D @ref specfem::sources::vector_source implementations</b>
 * - @ref specfem::sources::force<specfem::dimension::type::dim2>
 * - @ref specfem::sources::cosserat_force<specfem::dimension::type::dim2>
 * - @ref specfem::sources::adjoint_source<specfem::dimension::type::dim2>
 *
 * <b>3D @ref specfem::sources::vector_source implementations</b>
 * - @ref specfem::sources::force<specfem::dimension::type::dim3>
 *
 * <b>2D @ref specfem::sources::tensor_source implementations</b>
 * - @ref specfem::sources::moment_tensor< specfem::dimension::type::dim2 >
 *
 * See also
 * - @ref specfem::sources::source
 * - @ref specfem::sources::vector_source
 * - @ref specfem::sources::tensor_source
 */
namespace specfem::sources {

enum class source_type {
  vector_source, ///< Vector source
  tensor_source, ///< Tensor source
};

}

#include "source/source.hpp"
#include "source/source.tpp"
#include "source/tensor_source.hpp"
#include "source/vector_source.hpp"

/**
 * Need forward declaration of the main templates prior to specialization of
 * the dimension specific source implementations.
 */
namespace specfem::sources {

/**
 * @brief Moment tensor source
 *
 * This class represents a moment tensor source in the specified dimension.
 *
 * @tparam DimensionTag The dimension tag (`dim2` or `dim3`)
 */
template <specfem::dimension::type DimensionTag> class moment_tensor;

/**
 * @brief Force source
 *
 * This class represents a force source in the specified dimension.
 *
 * @tparam DimensionTag The dimension tag (`dim2` or `dim3`)
 */
template <specfem::dimension::type DimensionTag> class force;

/**
 * @brief Cosserat force source
 *
 * This class represents a Cosserat force source in the specified dimension.
 *
 * @tparam DimensionTag The dimension tag (`dim2` or `dim3`)
 */
template <specfem::dimension::type DimensionTag> class cosserat_force;

/**
 * @brief Adjoint source
 *
 * This class represents an adjoint source in the specified dimension.
 *
 * @tparam DimensionTag The dimension tag (`dim2` or `dim3`)
 */
template <specfem::dimension::type DimensionTag> class adjoint_source;

/**
 * @brief External source
 *
 * This class represents an external source in the specified dimension.
 *
 * @tparam DimensionTag The dimension tag (`dim2` or `dim3`)
 */
template <specfem::dimension::type DimensionTag> class external;

} // namespace specfem::sources

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
