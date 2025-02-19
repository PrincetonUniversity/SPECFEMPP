#include "jacobian/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "mesh/boundaries/acoustic_free_surface.hpp"
#include "mesh/control_nodes/control_nodes.hpp"
#include "mesh/elements/axial_elements.hpp"
#include "mesh/materials/materials.hpp"
#include "mesh/modifiers/modifiers.hpp"
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

static void subdivide_inherit_edgecond(
    const specfem::kokkos::HostView1d<
        specfem::mesh::materials::material_specification> &ispec_matspec,
    const specfem::mesh::modifiers<specfem::dimension::type::dim2> &modifiers,
    std::vector<int> &index_mapping,
    std::vector<specfem::enums::boundaries::type> &type, const int ispec_old,
    const specfem::enums::boundaries::type edge_type,
    const int ispec_old_to_new_offset) {
  int subz, subx;
  const auto mat = ispec_matspec(ispec_old);
  std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);

  // number of subdivided elements to set
  const int nsub = (edge_type == specfem::enums::boundaries::type::TOP ||
                    edge_type == specfem::enums::boundaries::type::BOTTOM)
                       ? subx
                       : subz;
  for (int isub = 0; isub < nsub; isub++) {
    int isubz, isubx;
    switch (edge_type) {
    case specfem::enums::boundaries::type::RIGHT:
      isubz = isub;
      isubx = subx - 1;
      break;
    case specfem::enums::boundaries::type::TOP:
      isubz = subz - 1;
      isubx = isub;
      break;
    case specfem::enums::boundaries::type::LEFT:
      isubz = subz - 1;
      isubx = isub;
      break;
    case specfem::enums::boundaries::type::BOTTOM:
      isubz = subz - 1;
      isubx = isub;
      break;
    default:
      throw std::runtime_error("Edge type NONE found in specfem::mesh::mesh.");
      break;
    }
    const int ispec_new = ispec_old_to_new_offset + (subx * isubz + isubx);
    index_mapping.push_back(ispec_new);
    type.push_back(edge_type);
  }
}

/**
 * @brief Computes the l^2 error squared between the two edges (difference in
 * knod positions). Positive orientation is assumed, and that the two elements
 * are different. Returns <err, parity>
 *
 * @param knods - struct of control node indices
 * @param coord - struct of control node coords
 * @param ispec1 - index of element 1
 * @param ispec2 - index of element 2
 * @param edge1 - edge of element 1 to check
 * @param edge2 - edge of element 2 to check
 */
static inline std::pair<type_real, bool>
edge_error(const Kokkos::View<int **, Kokkos::HostSpace> &knods,
           const Kokkos::View<type_real **, Kokkos::HostSpace> &coord,
           const int ispec1, const specfem::enums::edge::type edge1,
           const int ispec2, const specfem::enums::edge::type edge2) {
  const int ngnod = knods.extent(0);
  constexpr auto edge_rotation = [](const specfem::enums::edge::type edge) {
    switch (edge) {
    case specfem::enums::edge::RIGHT:
      return 0;
    case specfem::enums::edge::TOP:
      return 1;
    case specfem::enums::edge::LEFT:
      return 2;
    case specfem::enums::edge::BOTTOM:
      return 3;
    case specfem::enums::edge::NONE:
      return -1; // this should never be called
    }
  };
  /*     Computing parity:
   * To compare knod positions, we need to know which node corresponds on either
   * side. We could take the minimum among node reorderings (flip or not), but
   * if we are meant to check which edge is mating between ispec1 and ispec2,
   * there will always be one specific state of flipping or not.
   *
   * If two edges align, then the coordinate systems on either side may be in
   * the same or opposite directions. If we denote the direction from negative
   * to positive local coordinate as positive edge orientation, two edges of the
   * same type mating has parity one, since the 180 degree rotation of one
   * element flips the system. Similarly, if two edges of opposite sides mate,
   * the parity is zero, since the two elements can be joined without any
   * rotation.
   *
   * We can break this down into the following rule. If we index the edges as
   * above, where edge1 -> i and edge2 -> j, we can follow the table (using mod
   * 4 arithmetic): i    j-i    parity
   * -------------------
   *  i     0       1
   *  i     2       0
   *  0     1       0
   *  1     1       1
   *  2     1       0
   *  3     1       1
   *
   * These can be verified easily on paper, but this is equivalent to what we
   * have below
   */
  int erot1 = edge_rotation(edge1);
  int rotdif =
      edge_rotation(edge2) - erot1 + 4; // +4 since % doesn't "like" negatives.
  const bool parity =
      (rotdif % 2 == 0) ? (1 - rotdif / 2) : ((erot1 + rotdif / 2) % 2);
  constexpr auto get_firstlastnodes =
      [](int &low, int &high, const specfem::enums::edge::type edge) {
        // set low/high = ignod of low/high coordinate on edge (in terms of
        // local coordinates)
        switch (edge) {
        case specfem::enums::edge::NONE:
          low = high = 0;
          break;
        case specfem::enums::edge::TOP:
          low = 3;
          high = 2;
          break;
        case specfem::enums::edge::BOTTOM:
          low = 0;
          high = 1;
          break;
        case specfem::enums::edge::LEFT:
          low = 0;
          high = 3;
          break;
        case specfem::enums::edge::RIGHT:
          low = 1;
          high = 2;
          break;
        }
      };
  switch (ngnod) {
  case 4:
  case 9: {
    int inod1[2];
    int inod2[2];
    get_firstlastnodes(inod1[0], inod1[1], edge1);
    if (parity) {
      get_firstlastnodes(inod2[1], inod2[0], edge2);
    } else {
      get_firstlastnodes(inod2[0], inod2[1], edge2);
    }
    type_real err2 = 0;
    for (int i = 0; i < 2; i++) {
      inod1[i] = knods(inod1[i], ispec1);
      inod2[i] = knods(inod2[i], ispec2);
      // if control nodes are same, take the shortcut and say that error is zero
      if (inod1[i] != inod2[i]) {
        type_real tmp = coord(0, inod2[i]) - coord(0, inod1[i]);
        err2 += tmp * tmp;
        tmp = coord(1, inod2[i]) - coord(1, inod1[i]);
        err2 += tmp * tmp;
      }
    }
    return std::make_pair(err2, parity);
  }
  default:
    throw std::runtime_error("Invalid number of control nodes: " +
                             std::to_string(ngnod));
  }
}

template <specfem::element::medium_tag Medium1,
          specfem::element::medium_tag Medium2>
inline void subdivide_coupled_interface(
    specfem::mesh::interface_container<specfem::dimension::type::dim2, Medium1,
                                       Medium2> &interface,
    const std::vector<int> &ispec_old_to_new_offsets,
    const specfem::kokkos::HostView1d<
        specfem::mesh::materials::material_specification>
        &material_index_mapping_old,
    const specfem::mesh::modifiers<specfem::dimension::type::dim2> &modifiers,
    const Kokkos::View<int **, Kokkos::HostSpace> &knods_old,
    const Kokkos::View<type_real **, Kokkos::HostSpace> &coord_old) {
  constexpr std::array<specfem::enums::edge::type, 4> edge_enumeration = {
    specfem::enums::edge::type::RIGHT, specfem::enums::edge::type::TOP,
    specfem::enums::edge::type::BOTTOM, specfem::enums::edge::type::LEFT
  };

  std::vector<int> medium1_indices_new;
  std::vector<int> medium2_indices_new;

  int subz1, subx1, subz2, subx2;
  for (int ielem = 0; ielem < interface.num_interfaces; ielem++) {
    int ispec1 = interface.medium1_index_mapping(ielem);
    int ispec2 = interface.medium2_index_mapping(ielem);

    // recover which edges are joined: use an argmin approach.
    specfem::enums::edge::type edge1;
    specfem::enums::edge::type edge2;
    type_real err;
    bool parity;
    type_real min_err = std::numeric_limits<type_real>::max();
    for (int e1 = 0; e1 < 4; e1++) {
      specfem::enums::edge::type e1_ = edge_enumeration[e1];
      for (int e2 = 0; e2 < 4; e2++) {
        specfem::enums::edge::type e2_ = edge_enumeration[e2];
        std::tie(err, parity) =
            edge_error(knods_old, coord_old, ispec1, e1_, ispec2, e2_);
        if (err < min_err) {
          edge1 = e1_;
          edge2 = e2_;
          min_err = err;
        }
      }
    }

    // get the material subdivisions
    auto mat = material_index_mapping_old(ispec1);
    std::tie(subx1, subz1) = modifiers.get_subdivision(mat.database_index);
    mat = material_index_mapping_old(ispec2);
    std::tie(subx2, subz2) = modifiers.get_subdivision(mat.database_index);

    // subdivisions along mating edges
    bool horiz1 = edge1 == specfem::enums::edge::type::RIGHT ||
                  edge1 == specfem::enums::edge::type::LEFT;
    bool horiz2 = edge2 == specfem::enums::edge::type::RIGHT ||
                  edge2 == specfem::enums::edge::type::LEFT;
    int edgesub1 = horiz1 ? subz1 : subx1;
    int edgesub2 = horiz2 ? subz2 : subx2;

    // only include conforming edges.
    if (edgesub1 == edgesub2) {
      // store the non-varying subdivision index
      int isub_nonvar1 = (horiz1 ? subx1 : subz1) *
                         ((edge1 == specfem::enums::edge::type::RIGHT ||
                           edge1 == specfem::enums::edge::type::TOP)
                              ? 1
                              : 0);
      int isub_nonvar2 = (horiz2 ? subx2 : subz2) *
                         ((edge2 == specfem::enums::edge::type::RIGHT ||
                           edge2 == specfem::enums::edge::type::TOP)
                              ? 1
                              : 0);

      int isubx1 = isub_nonvar1, isubz1 = isub_nonvar1;
      int isubx2 = isub_nonvar2, isubz2 = isub_nonvar2;
      for (int isub = 0; isub < edgesub1; isub++) {
        if (horiz1) {
          isubz1 = isub;
        } else {
          isubx1 = isub;
        }
        if (horiz2) {
          isubz2 = isub;
        } else {
          isubx2 = isub;
        }
        medium1_indices_new.push_back(ispec_old_to_new_offsets[ispec1] +
                                      subx1 * isubz1 + isubx1);
        medium2_indices_new.push_back(ispec_old_to_new_offsets[ispec2] +
                                      subx2 * isubz2 + isubx2);
      }
    }

    interface =
        specfem::mesh::interface_container<specfem::dimension::type::dim2,
                                           Medium1, Medium2>(
            medium1_indices_new.size());
    for (int i = 0; i < interface.num_interfaces; i++) {
      interface.medium1_index_mapping(i) = medium1_indices_new[i];
      interface.medium2_index_mapping(i) = medium2_indices_new[i];
    }
  }
}

static void subdivide(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::mesh::modifiers<specfem::dimension::type::dim2> &modifiers) {
  constexpr auto DimensionType = specfem::dimension::type::dim2;
  const int nspec_old = mesh.nspec;
  const int ngnod = mesh.control_nodes.ngnod;
  const int ngnod_new = 9;
  const auto material_index_mapping_old = mesh.materials.material_index_mapping;

  /* we will expand coords, but we don't know to what size yet; use dynamic
   * resize of vector.
   */
  std::vector<std::pair<type_real, type_real> > coord_new(mesh.npgeo);
  // transfer coords to vector
  for (int i = 0; i < mesh.npgeo; i++) {
    coord_new[i] = std::make_pair(mesh.control_nodes.coord(0, i),
                                  mesh.control_nodes.coord(1, i));
  }

  // this is how we map from old ispec to new ispec
  std::vector<int> ispec_old_to_new_offsets(nspec_old);
  const auto ispec_old_to_new = [&](const int ispec, const int isubz,
                                    const int isubx) {
    const auto mat = material_index_mapping_old(ispec);
    return ispec_old_to_new_offsets[ispec] +
           std::get<0>(modifiers.get_subdivision(mat.database_index)) * isubz +
           isubx;
  };

  int nspec_new = 0; // update as we iterate
  int subz, subx;
  std::vector<int> knods_new;
  std::vector<specfem::mesh::materials::material_specification>
      material_index_mapping_new;
  for (int ispec = 0; ispec < nspec_old; ispec++) {
    const auto mat = material_index_mapping_old(ispec);
    std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);

    // form a matrix of node indices that will be used, making new ones where
    // necessary.
    const int ispec_ngnod_z = 2 * subz + 1;
    const int ispec_ngnod_x = 2 * subx + 1;
    const int total_knods = ispec_ngnod_x * ispec_ngnod_z;
    std::vector<int> ispec_knod(total_knods);
    for (int iz = 0; iz < ispec_ngnod_z; iz++) {
      for (int ix = 0; ix < ispec_ngnod_x; ix++) {
        const int ispec_inod = ispec_ngnod_x * iz + ix;
        if (iz == 0) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(0, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(1, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(4, ispec);
            continue;
          }
        } else if (iz == ispec_ngnod_z - 1) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(3, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(2, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(6, ispec);
            continue;
          }
        } else if (iz == subz && ngnod == 9) {
          if (ix == 0) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(7, ispec);
            continue;
          } else if (ix == ispec_ngnod_x - 1) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(5, ispec);
            continue;
          } else if (ix == subx && ngnod == 9) {
            ispec_knod[ispec_inod] = mesh.control_nodes.knods(8, ispec);
            continue;
          }
        }
        // nod does not already exist. Make it exist
        auto shape_functions = specfem::jacobian::define_shape_functions(
            ix / ((type_real)subx) - 1, iz / ((type_real)subz) - 1, ngnod);

        double xcor = 0.0;
        double zcor = 0.0;

        for (int in = 0; in < ngnod; in++) {
          const int knod = mesh.control_nodes.knods(in, ispec);
          xcor += mesh.control_nodes.coord(0, knod) * shape_functions[in];
          zcor += mesh.control_nodes.coord(1, knod) * shape_functions[in];
        }
        ispec_knod[ispec_inod] = coord_new.size();
        coord_new.push_back(std::make_pair(xcor, zcor));
      }
    }
    ispec_old_to_new_offsets[ispec] = nspec_new;
    for (int isubz = 0; isubz < subz; isubz++) {
      for (int isubx = 0; isubx < subx; isubx++) {
        // new element (based on new ngnod=9)
        const int inod = ispec_ngnod_x * (isubz * 2) + isubx * 2;
        knods_new.push_back(ispec_knod[inod]);
        knods_new.push_back(ispec_knod[inod + 2]);
        knods_new.push_back(ispec_knod[inod + 2 + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + 1]);
        knods_new.push_back(ispec_knod[inod + 2 + ispec_ngnod_x]);
        knods_new.push_back(ispec_knod[inod + 1 + ispec_ngnod_x * 2]);
        knods_new.push_back(ispec_knod[inod + ispec_ngnod_x]);
        knods_new.push_back(ispec_knod[inod + 1 + ispec_ngnod_x]);
        material_index_mapping_new.push_back(mat);
        nspec_new++;
      }
    }
  }

  // handle interfaces here, since we reference the old control nodes
  subdivide_coupled_interface(
      mesh.coupled_interfaces.elastic_acoustic, ispec_old_to_new_offsets,
      material_index_mapping_old, modifiers, mesh.control_nodes.knods,
      mesh.control_nodes.coord);
  subdivide_coupled_interface(
      mesh.coupled_interfaces.acoustic_poroelastic, ispec_old_to_new_offsets,
      material_index_mapping_old, modifiers, mesh.control_nodes.knods,
      mesh.control_nodes.coord);
  subdivide_coupled_interface(
      mesh.coupled_interfaces.elastic_poroelastic, ispec_old_to_new_offsets,
      material_index_mapping_old, modifiers, mesh.control_nodes.knods,
      mesh.control_nodes.coord);

  mesh.nspec = nspec_new;

  // finalize control nodes struct and materials struct
  const int npgeo_new = coord_new.size();
  mesh.npgeo = npgeo_new;
  mesh.materials.material_index_mapping = specfem::kokkos::HostView1d<
      specfem::mesh::materials::material_specification>(
      "specfem::mesh::material_index_mapping", nspec_new);
  mesh.control_nodes =
      specfem::mesh::control_nodes<specfem::dimension::type::dim2>(
          2, nspec_new, ngnod_new, coord_new.size());

  for (int igeo = 0; igeo < npgeo_new; igeo++) {
    std::tie(mesh.control_nodes.coord(0, igeo),
             mesh.control_nodes.coord(1, igeo)) = coord_new[igeo];
  }
  for (int ispec = 0; ispec < nspec_new; ispec++) {
    for (int inod = 0; inod < ngnod_new; inod++) {
      mesh.control_nodes.knods(inod, ispec) =
          knods_new[ispec * ngnod_new + inod];
    }
    mesh.materials.material_index_mapping(ispec) =
        material_index_mapping_new[ispec];
  }

  // boundaries
  { // absorbing
    std::vector<int> index_mapping;
    std::vector<specfem::enums::boundaries::type> type;
    auto &bdry = mesh.boundaries.absorbing_boundary;
    for (int ielem = 0; ielem < bdry.nelements; ielem++) {
      const int ispec = bdry.index_mapping(ielem);
      const specfem::enums::boundaries::type edge = bdry.type(ielem);
      subdivide_inherit_edgecond(material_index_mapping_old, modifiers,
                                 index_mapping, type, ispec, edge,
                                 ispec_old_to_new_offsets[ispec]);
    }
    bdry =
        specfem::mesh::absorbing_boundary<DimensionType>(index_mapping.size());
    for (int ielem = 0; ielem < bdry.nelements; ielem++) {
      bdry.index_mapping(ielem) = index_mapping[ielem];
      bdry.type(ielem) = type[ielem];
    }
  }
  { // acoustic free surface
    std::vector<int> index_mapping;
    std::vector<specfem::enums::boundaries::type> type;
    auto &bdry = mesh.boundaries.acoustic_free_surface;
    for (int ielem = 0; ielem < bdry.nelem_acoustic_surface; ielem++) {
      const int ispec = bdry.index_mapping(ielem);
      const specfem::enums::boundaries::type edge = bdry.type(ielem);
      subdivide_inherit_edgecond(material_index_mapping_old, modifiers,
                                 index_mapping, type, ispec, edge,
                                 ispec_old_to_new_offsets[ispec]);
    }
    bdry = specfem::mesh::acoustic_free_surface<DimensionType>(
        index_mapping.size());
    for (int ielem = 0; ielem < bdry.nelem_acoustic_surface; ielem++) {
      bdry.index_mapping(ielem) = index_mapping[ielem];
      bdry.type(ielem) = type[ielem];
    }
  }

  { // axial nodes
    auto &axi = mesh.axial_nodes;
    auto axiflag = axi.is_on_the_axis;
    axi = specfem::mesh::elements::axial_elements<DimensionType>(nspec_new);
    for (int ispec = 0; ispec < nspec_old; ispec++) {
      if (!axiflag(ispec)) {
        continue;
      }
      auto mat = material_index_mapping_old(ispec);
      std::tie(subx, subz) = modifiers.get_subdivision(mat.database_index);
      if (subz != 1 || subx != 1) {
        throw std::runtime_error(
            "Subdividing elements on the axis not yet supported");
      }
      axi.is_on_the_axis(ispec_old_to_new_offsets[ispec]) = true;
    }
  }

  // parameters
  mesh.parameters.ngnod = ngnod_new;
  mesh.parameters.nspec = nspec_new;
  mesh.parameters.nelemabs = mesh.boundaries.absorbing_boundary.nelements;
  mesh.parameters.nelem_acoustic_surface =
      mesh.boundaries.acoustic_free_surface.nelem_acoustic_surface;
  // mesh.parameters.nelem_acforcing = 0;
  mesh.parameters.num_fluid_solid_edges =
      mesh.coupled_interfaces.elastic_acoustic.num_interfaces;
  mesh.parameters.num_fluid_poro_edges =
      mesh.coupled_interfaces.acoustic_poroelastic.num_interfaces;
  mesh.parameters.num_solid_poro_edges =
      mesh.coupled_interfaces.elastic_poroelastic.num_interfaces;
  // mesh.parameters.nelem_on_the_axis = 0;
  // mesh.parameters.nnodes_tangential_curve = 0;

  mesh.tags = specfem::mesh::tags<specfem::dimension::type::dim2>(
      mesh.materials, mesh.boundaries);
}

template <>
void specfem::mesh::modifiers<specfem::dimension::type::dim2>::apply(
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh) const {
  subdivide(mesh, *this);

  // forcing boundary skipped
}
