#pragma once

#include "specfem/receivers.hpp"
#include <memory>
#include <vector>

/**
 * @brief Shared test receiver solutions for both file and YAML tests
 */

using ReceiverVector2DType = std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim2> > >;
using ReceiverVector3DType = std::vector<std::shared_ptr<
    specfem::receivers::receiver<specfem::dimension::type::dim3> > >;

// 2D receiver test solutions
extern const ReceiverVector2DType empty_receivers_2d;
extern const ReceiverVector2DType single_receiver_2d;
extern const ReceiverVector2DType two_receivers_2d;
extern const ReceiverVector2DType three_receivers_2d;
extern const ReceiverVector2DType ten_receivers_2d;

// 3D receiver test solutions
extern const ReceiverVector3DType empty_receivers_3d;
extern const ReceiverVector3DType single_receiver_3d;
extern const ReceiverVector3DType two_receivers_3d;
extern const ReceiverVector3DType three_receivers_3d;
