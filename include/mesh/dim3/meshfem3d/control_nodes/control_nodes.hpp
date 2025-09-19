#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh::meshfem3d {

template <specfem::dimension::type Dimension> struct ControlNodes {
public:
  constexpr static auto dimension_tag = Dimension;

private:
  constexpr static auto ndim = specfem::dimension::dimension<Dimension>::dim;
  using ViewType =
      Kokkos::View<type_real *[3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

public:
  ControlNodes() = default;
  ~ControlNodes() = default;

  ControlNodes(int nnodes_)
      : nnodes(nnodes_),
        coordinates("specfem::mesh::meshfem3d::ControlNodes", nnodes_) {}

  int nnodes;
  ViewType coordinates;
};

} // namespace specfem::mesh::meshfem3d
