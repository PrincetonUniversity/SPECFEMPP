#ifndef _BOUNDARIES_HPP
#define _BOUNDARIES_HPP

#include "absorbing_boundaries.hpp"
#include "acoustic_free_surface.hpp"
#include "forcing_boundaries.hpp"

namespace specfem {
namespace mesh {
struct boundaries {
  specfem::mesh::absorbing_boundary absorbing_boundary;
  specfem::mesh::acoustic_free_surface acoustic_free_surface;
  specfem::mesh::forcing_boundary forcing_boundary;

  boundaries() = default;
};
} // namespace mesh
} // namespace specfem

#endif
