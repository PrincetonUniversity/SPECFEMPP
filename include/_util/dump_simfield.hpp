#ifndef __UTIL_DUMP_DISCONT_SIMFIELD_
#define __UTIL_DUMP_DISCONT_SIMFIELD_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"

#include "adjacency_graph/adjacency_graph.hpp"

#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>

template <typename T, int dim, typename ViewType>
static void _stream_view(std::ofstream &stream, const ViewType &view) {
  T value;
  const char *val = (char *)&value;
  int extents[dim];
  for (int i = 0; i < dim; i++)
    extents[i] = view.extent(i);
  stream << "<" << typeid(T).name() << "(size=" << sizeof(T) << "B)>["
         << extents[0];
  for (int i = 1; i < dim; i++)
    stream << "," << extents[i];
  stream << "]";
  if constexpr (dim == 1) {
    for (int i = 0; i < extents[0]; i++) {
      value = view(i);
      stream.write(val, sizeof(T));
    }
  } else if constexpr (dim == 2) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        value = view(i, j);
        stream.write(val, sizeof(T));
      }
    }
  } else if constexpr (dim == 3) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        for (int k = 0; k < extents[2]; k++) {
          value = view(i, j, k);
          stream.write(val, sizeof(T));
        }
      }
    }
  } else if constexpr (dim == 4) {
    for (int i = 0; i < extents[0]; i++) {
      for (int j = 0; j < extents[1]; j++) {
        for (int k = 0; k < extents[2]; k++) {
          for (int l = 0; l < extents[3]; l++) {
            value = view(i, j, k, l);
            stream.write(val, sizeof(T));
          }
        }
      }
    }
  } else {
    static_assert(false, "dim not supported!");
  }
}

namespace _util {

static std::string tostr(specfem::element::boundary_tag tag) {
  switch (tag) {
  case specfem::element::boundary_tag::none:
    return "none";
  case specfem::element::boundary_tag::acoustic_free_surface:
    return "acoustic_free_surface";
  case specfem::element::boundary_tag::stacey:
    return "stacey";
  case specfem::element::boundary_tag::composite_stacey_dirichlet:
    return "composite_stacey_dirichlet";
  default:
    return "UNKNOWN";
  }
};

template <int num_sides>
void dump_adjacency_graph(
    const std::string &filename,
    const specfem::adjacency_graph::adjacency_graph<num_sides> &graph) {
  int nspec = graph.get_size();
  Kokkos::View<int *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      elems("_util::dump_adjacency_graph::elems", nspec);
  Kokkos::View<int8_t *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      sides("_util::dump_adjacency_graph::sides", nspec);
  Kokkos::View<bool *[num_sides], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      flips("_util::dump_adjacency_graph::flips", nspec);
  for (int i = 0; i < nspec; i++) {
    for (int side = 0; side < num_sides; side++) {
      specfem::adjacency_graph::adjacency_pointer adj =
          graph.get_adjacency(i, side);
      elems(i, side) = adj.elem;
      sides(i, side) = adj.side;
      flips(i, side) = adj.flip;
    }
  }
  std::ofstream dump;
  dump.open(filename);
  dump << "elem";
  _stream_view<int, 2>(dump, elems);
  dump << "side";
  _stream_view<int8_t, 2>(dump, sides);
  dump << "flips";
  _stream_view<bool, 2>(dump, flips);
  dump.close();
}

template <specfem::wavefield::type WavefieldType>
void dump_simfield(
    const std::string &filename,
    const specfem::compute::simulation_field<WavefieldType> &simfield,
    const specfem::compute::points &points) {
  std::ofstream dump;
  dump.open(filename);
  // dump points
  dump << "pts";
  _stream_view<type_real, 4>(dump, points.h_coord);
  dump << "index_mapping";
  _stream_view<int, 3>(dump, simfield.h_index_mapping);
  // //dump mesh adjacency
  // dump << "mesh_adj";
  // _stream_view<int,3>(dump,simfield.h_mesh_adjacency);
  // //dump ispec map
  // dump << "ispec_map";
  // _stream_view<int,2>(dump,simfield.h_assembly_ispec_mapping);
  // dump acoustic, elastic
  dump << "assembly_index_mapping";
  _stream_view<int, 2>(dump, simfield.h_assembly_index_mapping);
  dump << "acoustic_field";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_field);
  dump << "elastic_field";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_field);
  // //dump edge values
  // dump << "edge_values_x";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_x);
  // dump << "edge_values_z";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_z);

  dump.close();
}
template <specfem::wavefield::type WavefieldType>
void dump_simfield_per_step(
    const int istep, const std::string &filename,
    const specfem::compute::simulation_field<WavefieldType> &simfield,
    const specfem::compute::points &points) {
  dump_simfield(filename + std::to_string(istep) + ".dat", simfield, points);
}

void init_dirs(const boost::filesystem::path &dirname, bool clear = true) {
  if (clear) {
    boost::filesystem::remove_all(dirname);
  }
  boost::filesystem::create_directories(dirname);
}
} // namespace _util
#endif
