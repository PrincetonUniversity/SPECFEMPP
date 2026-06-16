#pragma once

#include "specfem/source.hpp"
#include <memory>
#include <vector>

/**
 * @brief Shared test source solutions for both file and YAML tests
 */

using SourceVector2DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>;
using SourceVector3DType = std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>;

// 2D source test solutions
extern const SourceVector2DType single_moment_tensor_2d;
extern const SourceVector2DType single_force_2d;
extern const SourceVector2DType single_cosserat_force_2d;
extern const SourceVector2DType multiple_sources_2d;

// 3D source test solutions
extern const SourceVector3DType single_force_3d;
extern const SourceVector3DType single_moment_tensor_3d;
extern const SourceVector3DType multiple_sources_3d;

// 3D CMTSOLUTION expected sources (GaussianHdur STF)
extern const SourceVector3DType single_moment_tensor_cmt_3d;
extern const SourceVector3DType spherical_moment_tensor_cmt_3d;
extern const SourceVector3DType single_moment_tensor_geographic_cmt_3d;

// 3D FORCESOLUTION expected sources (STF factor = 1.0)
extern const SourceVector3DType single_force_forcesolution_3d;

// 3D multi-source tests
extern const SourceVector3DType multiple_sources_cmt_3d;
extern const SourceVector3DType multiple_forces_forcesolution_3d;
