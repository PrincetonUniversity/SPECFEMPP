#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/mesh_entity.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

/**
 * @brief Integer codes of the SPECFEM3D_GLOBE thin mesh database.
 *
 * The thin database (see the record layout at the top of
 * @c fortran/meshfem3d_globe/meshfem3D/save_database_specfempp.F90) classifies
 * elements and faces with deliberately explicit integers:
 *
 * - region:   1 = crust/mantle, 2 = outer core, 3 = inner core
 * - medium:   1 = acoustic, 2 = elastic
 * - property: 0 = isotropic, 1 = anisotropic (includes TISO)
 * - entity:   faces 1-6, edges 7-18, corners 19-26 (the SPECFEM++ hexahedron
 *   numbering, so the values equal @c mesh_entity::dim3::type)
 *
 * Region conversions are shared with the evaluator boundary in
 * @c specfem/globe/region_codes.hpp. The remaining codes are private to the
 * database reader. Every function throws @c std::runtime_error naming the
 * offending value.
 */
namespace specfem::io::mesh::impl::fortran::dim3_globe {

/** @brief Medium tag for a database medium code. */
inline specfem::element::medium_tag to_medium_tag(const int code) {
  switch (code) {
  case 1:
    return specfem::element::medium_tag::acoustic;
  case 2:
    return specfem::element::medium_tag::elastic;
  default:
    throw std::runtime_error("Unknown globe medium code " +
                             std::to_string(code));
  }
}

/** @brief Property tag for a database property code. */
inline specfem::element::property_tag to_property_tag(const int code) {
  switch (code) {
  case 0:
    return specfem::element::property_tag::isotropic;
  case 1:
    return specfem::element::property_tag::anisotropic;
  default:
    throw std::runtime_error("Unknown globe property code " +
                             std::to_string(code));
  }
}

/** @brief Entity type (face, edge or corner) for a database entity code. */
inline specfem::mesh_entity::dim3::type to_entity(const int code) {
  if (code < static_cast<int>(specfem::mesh_entity::dim3::type::bottom) ||
      code >
          static_cast<int>(specfem::mesh_entity::dim3::type::top_back_right)) {
    throw std::runtime_error("Unknown globe entity code " +
                             std::to_string(code));
  }
  return static_cast<specfem::mesh_entity::dim3::type>(code);
}

/** @brief Face type for a database face code. */
inline specfem::mesh_entity::dim3::type to_face(const int code) {
  const auto entity = to_entity(code);
  const auto &faces = specfem::mesh_entity::dim3::faces;
  if (std::find(faces.begin(), faces.end(), entity) == faces.end()) {
    throw std::runtime_error("Globe entity code " + std::to_string(code) +
                             " is not a face");
  }
  return entity;
}

/** @brief Corner type for a database anchor-point code. */
inline specfem::mesh_entity::dim3::type to_anchor(const int code) {
  const auto entity = to_entity(code);
  const auto &corners = specfem::mesh_entity::dim3::corners;
  if (std::find(corners.begin(), corners.end(), entity) == corners.end()) {
    throw std::runtime_error("Globe entity code " + std::to_string(code) +
                             " is not a corner");
  }
  return entity;
}

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
