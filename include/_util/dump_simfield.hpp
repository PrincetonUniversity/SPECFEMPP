#ifndef __UTIL_DUMP_DISCONT_SIMFIELD_
#define __UTIL_DUMP_DISCONT_SIMFIELD_

#include "compute/fields/impl/field_impl.hpp"
#include "compute/fields/simulation_field.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"

#include "_util/edge_storages.hpp"
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

template <int ngllcapacity, int datacapacity>
void dump_edge_container(
    const std::string &filename,
    _util::edge_manager::edge_storage<ngllcapacity, datacapacity>
        &edge_storage) {
  int nedge = edge_storage.num_edges();
  int nintersect = edge_storage.num_intersections();

  Kokkos::View<int *[3], Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
      edge_intdat("_util::dump_edge_storage::edge_intdat", nedge);
  Kokkos::View<type_real *[ngllcapacity][2], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      pos("_util::dump_edge_storage::pos", nedge);
  Kokkos::View<type_real *[datacapacity][ngllcapacity], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      edgedata("_util::dump_edge_storage::edgedata", nedge);

  Kokkos::View<int *[5], Kokkos::LayoutLeft, specfem::kokkos::HostMemSpace>
      intersect_intdat("_util::dump_edge_storage::intersect_intdat",
                       nintersect);
  Kokkos::View<type_real *[5], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      intersect_floatdat("_util::dump_edge_storage::intersect_floatdat",
                         nintersect);
  Kokkos::View<type_real *[2][ngllcapacity][ngllcapacity], Kokkos::LayoutLeft,
               specfem::kokkos::HostMemSpace>
      mortar_trans("_util::dump_edge_storage::mortar_trans", nintersect);
  for (int i = 0; i < nedge; i++) {
    _util::edge_manager::edge_data<ngllcapacity, datacapacity> &edge_data =
        edge_storage.get_edge_on_host(i);
    edge_intdat(i, 0) = edge_data.parent.id;
    switch (edge_data.parent.bdry) {
    case specfem::enums::edge::type::RIGHT:
      edge_intdat(i, 1) = 0;
      break;
    case specfem::enums::edge::type::TOP:
      edge_intdat(i, 1) = 1;
      break;
    case specfem::enums::edge::type::LEFT:
      edge_intdat(i, 1) = 2;
      break;
    case specfem::enums::edge::type::BOTTOM:
      edge_intdat(i, 1) = 3;
      break;
    default:
      edge_intdat(i, 1) = -1;
      break;
    }
    edge_intdat(i, 2) = edge_data.ngll;
    for (int j = 0; j < edge_data.ngll; j++) {
      pos(i, j, 0) = edge_data.x[j];
      pos(i, j, 1) = edge_data.z[j];
      for (int k = 0; k < datacapacity; k++) {
        edgedata(i, k, j) = edge_data.data[k][j];
      }
    }
  }
  for (int i = 0; i < nintersect; i++) {
    _util::edge_manager::edge_intersection<ngllcapacity> &intersect =
        edge_storage.get_intersection_on_host(i);
    intersect_intdat(i, 0) = intersect.a_ref_ind;
    intersect_intdat(i, 1) = intersect.b_ref_ind;
    intersect_intdat(i, 2) = intersect.ngll;
    intersect_intdat(i, 3) = intersect.a_ngll;
    intersect_intdat(i, 4) = intersect.b_ngll;
    intersect_floatdat(i, 0) = intersect.a_param_start;
    intersect_floatdat(i, 1) = intersect.a_param_end;
    intersect_floatdat(i, 2) = intersect.b_param_start;
    intersect_floatdat(i, 3) = intersect.b_param_end;
    intersect_floatdat(i, 4) = intersect.relax_param;

    for (int j = 0; j < intersect.ngll; j++) {
      for (int k = 0; k < intersect.a_ngll; k++) {
        mortar_trans(i, 0, j, k) = intersect.a_mortar_trans[j][k];
      }
      for (int k = 0; k < intersect.b_ngll; k++) {
        mortar_trans(i, 1, j, k) = intersect.b_mortar_trans[j][k];
      }
    }
  }
  std::ofstream dump;
  dump.open(filename);
  dump << "edge_intdat";
  _stream_view<int, 2>(dump, edge_intdat);
  dump << "edge_pos";
  _stream_view<type_real, 3>(dump, pos);
  dump << "edge_data";
  _stream_view<type_real, 3>(dump, edgedata);
  dump << "intersect_intdat";
  _stream_view<int, 2>(dump, intersect_intdat);
  dump << "intersect_floatdat";
  _stream_view<type_real, 2>(dump, intersect_floatdat);
  dump << "intersect_mortartrans";
  _stream_view<type_real, 4>(dump, mortar_trans);
  dump << "intersect_data";
  _stream_view<type_real, 2>(dump,
                             edge_storage.get_intersection_data_on_host());
  dump.close();
}

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

template <specfem::wavefield::simulation_field WavefieldType>
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
  dump << "acoustic_field_ddot";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_field_dot_dot);
  dump << "elastic_field_ddot";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_field_dot_dot);
  dump << "acoustic_mass_inverse";
  _stream_view<type_real, 2>(dump, simfield.acoustic.h_mass_inverse);
  dump << "elastic_mass_inverse";
  _stream_view<type_real, 2>(dump, simfield.elastic.h_mass_inverse);
  // //dump edge values
  // dump << "edge_values_x";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_x);
  // dump << "edge_values_z";
  // _stream_view<type_real,4>(dump,simfield.h_edge_values_z);

  dump.close();
}
template <specfem::wavefield::simulation_field WavefieldType>
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
