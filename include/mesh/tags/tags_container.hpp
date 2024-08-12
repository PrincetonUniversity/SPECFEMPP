#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace mesh {
namespace impl {
struct tags_container {
  specfem::element::medium_tag medium_tag;
  specfem::element::property_tag property_tag;
  specfem::element::boundary_tag boundary_tag;
};
} // namespace impl
} // namespace mesh
} // namespace specfem
