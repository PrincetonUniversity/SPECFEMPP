#pragma once

namespace specfem::test_fixture {

/**
 * @brief Constructs a test mesh given a compile-time initializer.
 *
 * @tparam MeshInitializer2D initializer, of type
 * MeshInitializer2D::MeshInitializer2D
 */
template <typename MeshInitializer2D> struct Mesh2D;

/**
 * @brief Initializes Mesh2D, a mesh fixture
 *
 */
namespace MeshInitializer2D {
struct MeshInitializer2D {};
} // namespace MeshInitializer2D

/**
 * @brief Provides helper functions for an element (control nodes -> elements,
 * etc.)
 *
 */
template <typename MeshElementInit2D> struct MeshElement2D;
/**
 * @brief Gives the quadrature points for a particular rule.
 *
 */
namespace MeshElementType2D {
struct MeshElementType2D {};
} // namespace MeshElementType2D
} // namespace specfem::test_fixture

#include "mesh/element_2d.hpp"
#include "mesh/mesh.hpp"
