#pragma once

#include "specfem/data_access/accessor.hpp"

// all non-specific declarations for NCIs
#include "specfem/data_access/accessor.hpp"
namespace specfem::test_fixture {

/**
 * @brief Manages views of field values along an edge.
 *
 * @tparam Initializer initializer type: must inherit
 * EdgeFunctionInitializer2D::Base
 */
template <typename Initializer> struct EdgeFunction2D;
/**
 * @brief Initializes views of field values along an edge.
 *
 */
namespace EdgeFunctionInitializer2D {
struct EdgeFunctionInitializer2D {};
template <typename AnalyticalFunctionInitializer,
          typename EdgePointsInitializer>
struct FromAnalyticalFunction;
} // namespace EdgeFunctionInitializer2D

/**
 * @brief Manages views of the transfer function.
 *
 * @tparam Initializer initializer type: must inherit
 * TransferFunctionInitializer2D::Base
 */
template <typename Initializer> struct TransferFunction2D;
namespace TransferFunctionInitializer2D {
struct TransferFunctionInitializer2D {};
template <typename EdgeQuadratureInitializer,
          typename IntersectionQuadratureInitializer>
struct FromQuadratureRules;
} // namespace TransferFunctionInitializer2D

/**
 * @brief Manages views of field values along an intersection.
 *
 * @tparam Initializer initializer type: must inherit
 * IntersectionFunctionInitializer2D::IntersectionFunctionInitializer2D
 */
template <typename Initializer> struct IntersectionFunction2D;
/**
 * @brief Initializes views of field values along an edge.
 *
 */
namespace IntersectionFunctionInitializer2D {
struct IntersectionFunctionInitializer2D {};
} // namespace IntersectionFunctionInitializer2D

/**
 * @brief Manages the data of an intersection (everything but field values),
 * providing a patch for AccessorPack.
 *
 * @tparam Initializer initializer type: must inherit from
 * IntersectionDataInitializer2D
 * @tparam PackedTypes list of accessors to derive the AccessorPack.
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer,
          specfem::data_access::DataClassType... PackedTypes>
struct IntersectionDataPack2D;

/**
 * @brief Specifies a two-sided intersection (self and coupled). Each
 * IntersectionDataInitializer2D gives a transfer function for each side,
 * and the integration weights (with any potential Jacobian scaling) for
 * intersection integration.
 *
 */
namespace IntersectionDataInitializer2D {
struct IntersectionDataInitializer2D {};
} // namespace IntersectionDataInitializer2D

/**
 * @brief Baseline view for a nonconforming data accessor
 * (core/specfem/chunk_edge/nonconforming_interface.hpp)
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer,
          specfem::data_access::DataClassType DataClassType>
struct NonconformingAccessorPatch2D;

// =================================================================================================
// We may wish to move these to somewhere else in the future: these are not
// exclusive to NCIs

/**
 * @brief Provides helper functions for a quadrature rule (Lagrange polynomial
 * evaluation, weight computation, etc.)
 *
 */
template <typename QuadraturePoints> struct QuadratureRule;
/**
 * @brief Gives the quadrature points for a particular rule.
 *
 */
namespace QuadraturePoints {
struct QuadraturePoints {};
} // namespace QuadraturePoints

/**
 * @brief Manages 1-parameter functions. These can be used, say for
 * edge-coordinate analytical fields
 */
namespace AnalyticalFunctionType {
struct AnalyticalFunctionType {};
} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
