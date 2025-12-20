#pragma once

#include "enumerations/connections.hpp"
// all non-specific declarations for NCIs
namespace specfem::test::fixture {

/**
 * @brief Manages views of field values along an edge.
 *
 * @tparam Initializer initializer type: must inherit
 * EdgeFunctionInitializer2D::Base
 */
template <typename Initializer, typename... BaseTypes> struct EdgeFunction2D;
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
 * @brief Manages views of field values along an intersection. (TODO: we may
 * wish to merge this with EdgeFunction2D)
 *
 * @tparam Initializer initializer type: must inherit
 * IntersectionFunction2D::IntersectionFunction2D
 */
template <typename Initializer> struct IntersectionFunction2D;
/**
 * @brief Initializes views of field values along an edge.
 *
 */
namespace IntersectionFunctionInitializer2D {
struct IntersectionFunctionInitializer2D {};
} // namespace IntersectionFunctionInitializer2D

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
namespace AnalyticalFunctionType1D {
struct AnalyticalFunctionType1D {};
} // namespace AnalyticalFunctionType1D

} // namespace specfem::test::fixture
