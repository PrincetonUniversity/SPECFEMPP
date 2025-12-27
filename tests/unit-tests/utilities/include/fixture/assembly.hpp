#pragma once

namespace specfem::test_fixture {

/**
 * @brief Constructs a test assembly given a compile-time initializer.
 *
 * @tparam AssemblyInitializer2D initializer, of type
 * MeshInitializer2D::MeshInitializer2D
 */
template <typename AssemblyInitializer2D> struct Assembly2D;

/**
 * @brief Initializes Assembly2D, an assembly fixture
 *
 */
namespace AssemblyInitializer2D {
struct AssemblyInitializer2D {};
} // namespace AssemblyInitializer2D

} // namespace specfem::test_fixture

#include "assembly/assembly_2d.hpp"
