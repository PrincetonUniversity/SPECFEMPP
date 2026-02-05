#include "specfem/enums.hpp"

std::string specfem::display::to_string(const specfem::display::format &fmt) {
  switch (fmt) {
  case specfem::display::format::PNG:
    return "PNG";
  case specfem::display::format::JPG:
    return "JPG";
  case specfem::display::format::on_screen:
    return "on_screen";
  case specfem::display::format::vtkhdf:
    return "vtkhdf";
  default:
    return "unknown";
  }
}

std::string
specfem::display::to_string(const specfem::display::component &comp) {
  switch (comp) {
  case specfem::display::component::x:
    return "x";
  case specfem::display::component::y:
    return "y";
  case specfem::display::component::z:
    return "z";
  case specfem::display::component::magnitude:
    return "magnitude";
  default:
    return "unknown";
  }
}
