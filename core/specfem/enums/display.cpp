#include "specfem/enums.hpp"

std::string
specfem::enums::to_string(const specfem::enums::display_format &fmt) {
  switch (fmt) {
  case specfem::enums::display_format::PNG:
    return "PNG";
  case specfem::enums::display_format::JPG:
    return "JPG";
  case specfem::enums::display_format::on_screen:
    return "on_screen";
  case specfem::enums::display_format::vtkhdf:
    return "vtkhdf";
  default:
    return "unknown";
  }
}

std::string
specfem::enums::to_string(const specfem::enums::display_component &comp) {
  switch (comp) {
  case specfem::enums::display_component::x:
    return "x";
  case specfem::enums::display_component::y:
    return "y";
  case specfem::enums::display_component::z:
    return "z";
  case specfem::enums::display_component::magnitude:
    return "magnitude";
  default:
    return "unknown";
  }
}
