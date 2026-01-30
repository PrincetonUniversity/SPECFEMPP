#pragma once

#include "enumerations/specfem_enums.hpp"
#include "specfem/medium_container.hpp"
#include "materials.hpp"
#include <variant>
#include <vector>

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
          specfem::mesh::materials<specfem::dimension::type::dim2>::material<type, property>::material(
    const int n_materials,
    const std::vector<specfem::medium_container::material<dimension_tag, type, property> >
        &l_materials)
    : n_materials(n_materials),
    element_materials(l_materials) {}
