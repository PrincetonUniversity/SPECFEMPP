#include "specfem/enums/coordinate_frame.hpp"

const std::string
specfem::enums::to_string(const specfem::enums::coordinate_frame &frame) {
  switch (frame) {
  case specfem::enums::coordinate_frame::cartesian:
    return "cartesian";
  case specfem::enums::coordinate_frame::geographic:
    return "geographic";
  default:
    return "unknown";
  }
}
