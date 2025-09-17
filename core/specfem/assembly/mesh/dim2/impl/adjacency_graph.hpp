#pragma once

#include "mesh/mesh.hpp"

namespace specfem::assembly::mesh_impl {

template <>
class adjacency_graph<specfem::dimension::type::dim2>
    : public specfem::mesh::adjacency_graph<specfem::dimension::type::dim2> {

private:
  using base_type =
      specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>;

public:
  using base_type::base_type;
};

} // namespace specfem::assembly::mesh_impl
