#pragma once

#include "enumerations/specfem_enums.hpp"
#include "medium/material.hpp"
#include "materials.hpp"
#include <variant>
#include <vector>

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
specfem::mesh::materials::material<type, property>::material(
    const int n_materials,
    const std::vector<specfem::medium::material<type, property> >
        &l_materials)
    : n_materials(n_materials),
      material_properties(l_materials) {}
