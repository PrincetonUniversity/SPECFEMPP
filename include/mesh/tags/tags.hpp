#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "mesh/materials/materials.hpp"
#include "tags_container.hpp"

namespace specfem {
namespace mesh {
struct tags {
  int nspec;
  specfem::kokkos::HostView1d<specfem::mesh::impl::tags_container>
      tags_container;

  tags(const int nspec) : tags_container("specfem::mesh::tags::tags", nspec) {}

  tags() = default;

  tags(const specfem::mesh::materials &materials,
       const specfem::mesh::boundaries &boundaries);
};
} // namespace mesh
} // namespace specfem
