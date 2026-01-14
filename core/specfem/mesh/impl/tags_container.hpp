#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace mesh {
namespace impl {
/**
 * @brief Struct to store tags for every spectral element
 *
 */
struct tags_container {
  specfem::element::medium_tag medium_tag;     ///< Medium tag
  specfem::element::property_tag property_tag; ///< Property tag
  specfem::element::boundary_tag boundary_tag; ///< Boundary tag

  tags_container(const specfem::element::medium_tag medium_tag_,
                 const specfem::element::property_tag property_tag_,
                 const specfem::element::boundary_tag boundary_tag_)
      : medium_tag(medium_tag_), property_tag(property_tag_),
        boundary_tag(boundary_tag_) {}

  tags_container() = default;
};
} // namespace impl
} // namespace mesh
} // namespace specfem
