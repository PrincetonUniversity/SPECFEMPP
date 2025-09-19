#pragma once

#include "specfem/source.hpp"
#include <memory>
#include <vector>

/**
 * @brief Shared test source solutions for both file and YAML tests
 */

using SourceVector2DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::dimension::type::dim2> > >;
using SourceVector3DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::dimension::type::dim3> > >;

// 2D source test solutions
extern const SourceVector2DType single_moment_tensor_2d;
extern const SourceVector2DType single_force_2d;
extern const SourceVector2DType single_cosserat_force_2d;
extern const SourceVector2DType multiple_sources_2d;

// 3D source test solutions
extern const SourceVector3DType single_force_3d;
extern const SourceVector3DType single_moment_tensor_3d;
extern const SourceVector3DType multiple_sources_3d;
