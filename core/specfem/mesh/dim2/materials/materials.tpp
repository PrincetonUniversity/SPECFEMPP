#pragma once

#include "specfem/enums.hpp"
#include "materials.hpp"
#include "specfem/medium_container.hpp"
#include <variant>
#include <vector>

template <specfem::element::medium_tag type,
          specfem::element::property_tag property,
          specfem::element::attenuation_tag attenuation>
specfem::mesh::materials<specfem::element::dimension_tag::dim2>::
    material<type, property, attenuation>::material(
        const int n_materials,
        const std::vector<specfem::medium_container::material<
            dimension_tag, type, property, attenuation> > &l_materials)
    : n_materials(n_materials), element_materials(l_materials) {}
