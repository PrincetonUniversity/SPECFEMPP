#pragma once

#include "../mesh.hpp"
#include "enumerations/dimension.hpp"
#include "specfem/point.hpp"

namespace specfem::test_fixture {

template <typename MeshElementInit2D> struct MeshElement2D {
  static_assert(std::is_base_of_v<MeshElementType2D::MeshElementType2D,
                                  MeshElementInit2D>,
                "MeshElement2D template parameter expects MeshElementType2D!");
};

namespace MeshElementType2D {
constexpr specfem::dimension::type DimensionTag =
    specfem::dimension::type::dim2;
constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

/**
 * @brief ngnod = 4 reference element [-1,1]^2
 *
 */
struct Reference {
  static constexpr int ngnod = 4;
  static constexpr std::array<std::array<type_real, ndim>, 4> control_nodes = {
    std::array<type_real, ndim>{ -1, -1 }, std::array<type_real, ndim>{ 1, -1 },
    std::array<type_real, ndim>{ 1, 1 }, std::array<type_real, ndim>{ -1, 1 }
  };
  static std::string description() { return "ngnod = 4 reference element"; }
};
} // namespace MeshElementType2D
} // namespace specfem::test_fixture
