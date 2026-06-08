#include "specfem/coordinate_systems/cartesian.hpp"

#include "specfem/utilities/is_close.hpp"

#include <sstream>

// ── 2D ──────────────────────────────────────────────────────────────────────

bool specfem::coordinate_systems::
    cartesian_coordinates<specfem::element::dimension_tag::dim2>::operator==(
        const specfem::coordinate_systems::coordinates<
            specfem::element::dimension_tag::dim2> &other) const {
  const auto *o =
      dynamic_cast<const specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim2> *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(x, o->x) &&
         specfem::utilities::is_close(z, o->z);
}

std::string specfem::coordinate_systems::cartesian_coordinates<
    specfem::element::dimension_tag::dim2>::print() const {
  std::ostringstream os;
  os << "cartesian_2d(x=" << x << ", z=" << z;
  if (origin.has_value()) {
    os << ", origin=[" << (*origin)[0] << ", " << (*origin)[1] << "]";
  } else {
    os << ", origin=nullopt";
  }
  os << ")";
  return os.str();
}

// ── 3D ──────────────────────────────────────────────────────────────────────

bool specfem::coordinate_systems::
    cartesian_coordinates<specfem::element::dimension_tag::dim3>::operator==(
        const specfem::coordinate_systems::coordinates<
            specfem::element::dimension_tag::dim3> &other) const {
  const auto *o =
      dynamic_cast<const specfem::coordinate_systems::cartesian_coordinates<
          specfem::element::dimension_tag::dim3> *>(&other);
  if (!o)
    return false;
  return specfem::utilities::is_close(x, o->x) &&
         specfem::utilities::is_close(y, o->y) &&
         specfem::utilities::is_close(z, o->z);
}

std::string specfem::coordinate_systems::cartesian_coordinates<
    specfem::element::dimension_tag::dim3>::print() const {
  std::ostringstream os;
  os << "cartesian_3d(x=" << x << ", y=" << y << ", z=" << z;
  if (origin.has_value()) {
    os << ", origin=[" << (*origin)[0] << ", " << (*origin)[1] << ", "
       << (*origin)[2] << "]";
  } else {
    os << ", origin=nullopt";
  }
  os << ")";
  return os.str();
}
