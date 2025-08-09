#pragma once
#include "enumerations/dimension.hpp"
#define CORE_SPECFEM_ASSEMBLY_NONCONFORMINGINTERFACES_INTERFACEMODULES_HPP

namespace specfem::assembly::interface {

struct initializer {
  int num_medium1_edges;
  int num_medium2_edges;
  int num_interfaces;
  int nquad_medium1;
  int nquad_medium2;
  int nquad_interface;
  initializer(int num_medium1_edges, int num_medium2_edges, int num_interfaces,
              int nquad_medium1, int nquad_medium2, int nquad_interface)
      : num_medium1_edges(num_medium1_edges),
        num_medium2_edges(num_medium2_edges), num_interfaces(num_interfaces),
        nquad_medium1(nquad_medium1), nquad_medium2(nquad_medium2),
        nquad_interface(nquad_interface) {};
  initializer() : initializer(0, 0, 0, 0, 0, 0) {};
};

namespace module {
enum class type { SINGLE_EDGE_CONTAINER, DOUBLE_EDGE_CONTAINER };

template <specfem::dimension::type DimensionType, type... modules>
struct container_args;

} // namespace module

} // namespace specfem::assembly::interface
