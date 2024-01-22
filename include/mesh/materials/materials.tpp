#ifndef _MESH_MATERIALS_TPP
#define _MESH_MATERIALS_TPP

#include "material/material.hpp"
#include "materials.hpp"
#include "enumerations/specfem_enums.hpp"
#include <vector>
#include <variant>

template <specfem::enums::element::type type,
          specfem::enums::element::property_tag property>
specfem::mesh::materials::material<type, property>::material(const int n_materials, const std::vector<specfem::material::material<type, property> > &l_materials)
    : n_materials(n_materials), material_properties("specfem::mesh::materials::material", n_materials) {

  for (int i = 0; i < n_materials; i++) {
    material_properties(i) = l_materials[i];
  }

  return;
}

#endif /* end of include guard: _MESH_MATERIALS_TPP */
