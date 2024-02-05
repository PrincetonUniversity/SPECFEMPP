#include "enumerations/specfem_enums.hpp"
#include "macros.hpp"
#include "point/boundary.hpp"

void specfem::point::boundary::update_boundary(
    const specfem::enums::boundaries::type &type,
    const specfem::enums::element::boundary_tag &tag) {
  if (type == specfem::enums::boundaries::type::TOP) {
    top = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM) {
    bottom = tag;
  } else if (type == specfem::enums::boundaries::type::LEFT) {
    left = tag;
  } else if (type == specfem::enums::boundaries::type::RIGHT) {
    right = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_RIGHT) {
    bottom_right = tag;
  } else if (type == specfem::enums::boundaries::type::BOTTOM_LEFT) {
    bottom_left = tag;
  } else if (type == specfem::enums::boundaries::type::TOP_RIGHT) {
    top_right = tag;
  } else if (type == specfem::enums::boundaries::type::TOP_LEFT) {
    top_left = tag;
  } else {
    ASSERT(false, "Error: Unknown boundary type");
  }
}

KOKKOS_FUNCTION
bool specfem::point::is_on_boundary(
    const specfem::enums::element::boundary_tag &tag,
    const specfem::point::boundary &type, const int &iz, const int &ix,
    const int &ngllz, const int &ngllx) {

  return (type.top == tag && iz == ngllz - 1) ||
         (type.bottom == tag && iz == 0) || (type.left == tag && ix == 0) ||
         (type.right == tag && ix == ngllx - 1) ||
         (type.bottom_right == tag && iz == 0 && ix == ngllx - 1) ||
         (type.bottom_left == tag && iz == 0 && ix == 0) ||
         (type.top_right == tag && iz == ngllz - 1 && ix == ngllx - 1) ||
         (type.top_left == tag && iz == ngllz - 1 && ix == 0);
}
