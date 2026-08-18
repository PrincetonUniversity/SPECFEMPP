#include "test_receiver_solutions.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include <memory>

// 2D receiver test solutions
const ReceiverVector2DType empty_receivers_2d = {};

const ReceiverVector2DType single_receiver_2d = { std::make_shared<
    specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
    "AA", "S0001", 300.0, 3000.0, 0.0) };

const ReceiverVector2DType two_receivers_2d = {
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0001", 300.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0002", 640.0, 3000.0, 0.0)
};

const ReceiverVector2DType three_receivers_2d = {
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0001", 300.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0002", 640.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0003", 980.0, 3000.0, 0.0)
};

const ReceiverVector2DType ten_receivers_2d = {
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0001", 300.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0002", 640.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0003", 980.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0004", 1320.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0005", 1660.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0006", 2000.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0007", 2340.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0008", 2680.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0009", 3020.0, 3000.0, 0.0),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim2>>(
      "AA", "S0010", 3360.0, 3000.0, 0.0)
};

// 3D receiver test solutions. Cartesian receivers defer resolution via
// read_coordinates_, so the expected receivers use cartesian_coordinates
// (default origin {0,0,0}) to match what the parser builds (compared by
// receiver::operator==, which compares read_coordinates_).
using cartesian_coordinates_3d_type =
    specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3>;

const ReceiverVector3DType empty_receivers_3d = {};

const ReceiverVector3DType single_receiver_3d = { std::make_shared<
    specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
    "AA", "S0001",
    std::make_unique<cartesian_coordinates_3d_type>(300.0, 3000.0, 2000.0)) };

const ReceiverVector3DType two_receivers_3d = {
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
      "AA", "S0001",
      std::make_unique<cartesian_coordinates_3d_type>(300.0, 3000.0, 2000.0)),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
      "AA", "S0002",
      std::make_unique<cartesian_coordinates_3d_type>(640.0, 3000.0, 2000.0))
};

const ReceiverVector3DType three_receivers_3d = {
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
      "AA", "S0001",
      std::make_unique<cartesian_coordinates_3d_type>(300.0, 3000.0, 2000.0)),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
      "AA", "S0002",
      std::make_unique<cartesian_coordinates_3d_type>(640.0, 3000.0, 2000.0)),
  std::make_shared<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>(
      "AA", "S0003",
      std::make_unique<cartesian_coordinates_3d_type>(980.0, 3000.0, 2000.0))
};
