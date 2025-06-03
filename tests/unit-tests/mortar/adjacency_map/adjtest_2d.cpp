#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "../../MPI_environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/interface.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

void test_assembly_mapping(
    specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
        &adjacencies,
    const int nspec, const int ngll = 5);

void run_test_conforming(std::string databasename) {
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto mesh =
      specfem::io::read_2d_mesh(databasename, specfem::enums::elastic_wave::psv,
                                specfem::enums::electromagnetic_wave::te, mpi);

  mpi->cout("Mesh read. Forming adjacency map.");
  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
      &adjacencies = mesh.adjacency_map;
  if (!adjacencies.was_initialized()) {
    adjacencies = specfem::mesh::adjacency_map::adjacency_map<
        specfem::dimension::type::dim2>(mesh);
  }
  std::ostringstream msg;
  msg << "Adjacency map formed. Comparing results...";
  mpi->cout(msg.str());

  // print out adjacencies
  msg = std::ostringstream();
#define COLWIDTH (7)
#define NUMCOLS (5)
#define NUM_DIGITS(st) (st < 10 ? 1 : (st < 100 ? 2 : (st < 1000 ? 3 : 4)))
  char entry[COLWIDTH];
  const auto print_entry = [&](bool terminate = false) {
    int stsize;
    for (stsize = 0; stsize < COLWIDTH && entry[stsize] != '\0'; stsize++) {
    }
    int padsize = COLWIDTH - stsize;
    for (int i = padsize / 2; i > 0; i--) {
      msg << ' ';
    }
    msg.write(entry, stsize);
    for (int i = padsize - padsize / 2; i > 0; i--) {
      msg << ' ';
    }
    if (terminate) {
      msg << '\n';
    } else {
      msg << '|';
    }
  };

  const auto set_entry_from_adj = [&](const int ispec,
                                      const specfem::enums::edge::type type) {
    if (adjacencies.has_conforming_adjacency(ispec, type)) {
      int ispec_adj;
      specfem::enums::edge::type type_adj;
      std::tie(ispec_adj, type_adj) =
          adjacencies.get_conforming_adjacency(ispec, type);
      int padding_size = NUM_DIGITS(ispec_adj);
      std::sprintf(
          entry, "%d%c", ispec_adj,
          type_adj == specfem::enums::edge::type::TOP
              ? 'T'
              : (type_adj == specfem::enums::edge::type::BOTTOM
                     ? 'B'
                     : (type_adj == specfem::enums::edge::type::LEFT ? 'L'
                                                                     : 'R')));
    } else if (adjacencies.has_boundary(ispec, type)) {
      std::sprintf(entry, "(bdry)");
    } else {
      std::sprintf(entry, "MTR");
    }
  };
  std::sprintf(entry, "ISPEC");
  print_entry();
  std::sprintf(entry, "RIGHT");
  print_entry();
  std::sprintf(entry, "TOP");
  print_entry();
  std::sprintf(entry, "LEFT");
  print_entry();
  std::sprintf(entry, "BOTTOM");
  print_entry(true);
  for (int i = 0; i < mesh.nspec; i++) {
    std::sprintf(entry, "%d", i);
    print_entry();
    set_entry_from_adj(i, specfem::enums::edge::type::RIGHT);
    print_entry();
    set_entry_from_adj(i, specfem::enums::edge::type::TOP);
    print_entry();
    set_entry_from_adj(i, specfem::enums::edge::type::LEFT);
    print_entry();
    set_entry_from_adj(i, specfem::enums::edge::type::BOTTOM);
    print_entry(true);
  }
  mpi->cout(msg.str());

  test_assembly_mapping(adjacencies, mesh.nspec);
}

void test_assembly_mapping(
    specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
        &adjacencies,
    const int nspec, const int ngll) {
  /*
   * To test adjacency_map::generate_assembly_mapping, we want to verify that
   * the mapping is valid. We will enforce these rules:
   *
   * - 1) Every node i has a global index 0 <= ind[i] < nglob. ind is
   * surjective.
   * - 2) ind[i] == ind[j] iff. i,j share a conforming edge or corner.
   *
   * Ordering is not constrained, as it can be changed by a simple renumbering.
   * We also assume that the sharing check is an equivalence relation.
   * Particularly, adjacencies.get_all_conforming_adjacencies() returns the same
   * set for other values in that set.
   */
  const auto assembly_out = adjacencies.generate_assembly_mapping(ngll);
  const auto index_mapping = assembly_out.first;
  const int nglob = assembly_out.second;

  // preimage of index_mapping.
  std::vector<std::vector<std::tuple<int, int, int> > > ind_to_nodes(
      nglob, std::vector<std::tuple<int, int, int> >());
  for (int ispec = 0; ispec < nspec; ispec++) {
    for (int ix = 0; ix < ngll; ix++) {
      for (int iz = 0; iz < ngll; iz++) {
        int ind = index_mapping(ispec, iz, ix);

        assert(0 <= ind && ind < nglob);
        ind_to_nodes[ind].push_back(std::make_tuple(ispec, iz, ix));
      }
    }
  }

  // is each index used?
  for (int iglob = 0; iglob < nglob; iglob++) {
    assert(!ind_to_nodes[iglob].empty());
  }

  // (1) passed. Loop again to verify each partition is equivalent

  const auto local_to_bdry = [&](const int ix, const int iz) {
    if (ix == 0) {
      if (iz == 0) {
        return specfem::enums::boundaries::type::BOTTOM_LEFT;
      } else if (iz == ngll - 1) {
        return specfem::enums::boundaries::type::TOP_LEFT;
      } else {
        return specfem::enums::boundaries::type::LEFT;
      }
    } else if (ix == ngll - 1) {
      if (iz == 0) {
        return specfem::enums::boundaries::type::BOTTOM_RIGHT;
      } else if (iz == ngll - 1) {
        return specfem::enums::boundaries::type::TOP_RIGHT;
      } else {
        return specfem::enums::boundaries::type::RIGHT;
      }
    } else {
      if (iz == 0) {
        return specfem::enums::boundaries::type::BOTTOM;
      } else if (iz == ngll - 1) {
        return specfem::enums::boundaries::type::TOP;
      } else {
        assert(1 == 0);
        return specfem::enums::boundaries::type::RIGHT;
      }
    }
  };

  int ispec, iz, ix;
  for (int iglob = 0; iglob < nglob; iglob++) {
    const auto nodeset = ind_to_nodes[iglob];
    std::tie(ispec, iz, ix) = nodeset[0];
    if (0 < iz && iz < ngll - 1 && 0 < ix && ix < ngll - 1) {
      // interior. No adjacencies:
      assert(nodeset.size() == 1);
    } else {
      // check sets are "equal" (ix,iz) ~ bdry
      auto adjset = adjacencies.get_all_conforming_adjacencies(
          ispec, local_to_bdry(ix, iz));
      std::vector<std::pair<int, specfem::enums::boundaries::type> >
          nodeset_match(
              nodeset.size(),
              std::make_pair(-1, specfem::enums::boundaries::type::RIGHT));
      bool matchfail = false;
      // node and adjset are in 1-1 correspondence?

      for (int i = 0; i < nodeset.size(); i++) {
        const auto &node = nodeset[i];
        const auto it = adjset.find(std::make_pair(
            std::get<0>(node),
            local_to_bdry(std::get<2>(node), std::get<1>(node))));
        if (it == adjset.end()) {
          matchfail = true;
        } else {
          nodeset_match[i] = *it;
          adjset.erase(it);
        }
      }
      // none left.
      if (matchfail || !adjset.empty()) {
        std::ostringstream corrprint;
        corrprint << "Nodes sharing global index " << iglob
                  << " -> adjacencies\n"
                  << "   ispec   | ix | iz | paired boundary\n";
        const auto bd_to_string = [](specfem::enums::boundaries::type bd) {
          switch (bd) {
          case specfem::enums::boundaries::TOP_LEFT:
            return "TOP_LEFT";
          case specfem::enums::boundaries::TOP_RIGHT:
            return "TOP_RIGHT";
          case specfem::enums::boundaries::BOTTOM_LEFT:
            return "BOTTOM_LEFT";
          case specfem::enums::boundaries::BOTTOM_RIGHT:
            return "BOTTOM_RIGHT";
          case specfem::enums::boundaries::TOP:
            return "TOP";
          case specfem::enums::boundaries::LEFT:
            return "LEFT";
          case specfem::enums::boundaries::RIGHT:
            return "RIGHT";
          case specfem::enums::boundaries::BOTTOM:
            return "BOTTOM";
          default:
            return "";
          }
        };
        for (int i = 0; i < nodeset.size(); i++) {
          corrprint << std::setw(10) << std::get<0>(nodeset[i]) << " | "
                    << std::setw(2) << std::get<2>(nodeset[i]) << " | "
                    << std::setw(2) << std::get<1>(nodeset[i]) << " | ";
          if (nodeset_match[i].first != -1) {
            corrprint << bd_to_string(nodeset_match[i].second) << "\n";
          } else {
            corrprint << "\033[0;31mUNPAIRED\033[0m\n";
          }
        }
        if (!adjset.empty()) {
          corrprint << "\033[0;31mUnparied boundaries (from adjacency map):";
          for (const auto &unpaired : adjset) {
            corrprint << "\n " << bd_to_string(unpaired.second)
                      << " @ ispec = " << unpaired.first;
            if (unpaired.second == specfem::enums::boundaries::BOTTOM_RIGHT) {
              corrprint << " (corner iglob = "
                        << index_mapping(unpaired.first, 0, ngll - 1) << ")";
            } else if (unpaired.second ==
                       specfem::enums::boundaries::TOP_RIGHT) {
              corrprint << " (corner iglob = "
                        << index_mapping(unpaired.first, ngll - 1, ngll - 1)
                        << ")";
            } else if (unpaired.second ==
                       specfem::enums::boundaries::TOP_LEFT) {
              corrprint << " (corner iglob = "
                        << index_mapping(unpaired.first, ngll - 1, 0) << ")";
            } else if (unpaired.second ==
                       specfem::enums::boundaries::BOTTOM_LEFT) {
              corrprint << " (corner iglob = "
                        << index_mapping(unpaired.first, 0, 0) << ")";
            }
          }
          corrprint << "\033[0m\n";
        }
        {
          corrprint << "No unpaired adjacency-map boundaries (as desired).";
        }
        FAIL() << "--------------------------------------------------\n"
               << "\033[0;31m[FAILED]\033[0m Test failed\n"
               << "               test_assembly_mapping\n"
               << "             (Global index " << iglob << ")\n"
               << " - Correspondence between adjacency and iglob\n"
               << " - adjacencies built from:\n"
               << "       ispec = " << ispec << "\n"
               << "          ix = " << ix << "\n"
               << "          iz = " << iz << "\n"
               << "    boundary = " << bd_to_string(local_to_bdry(ix, iz))
               << "\n"
               << "--------------------------------------------------\n"
               << corrprint.str() << std::endl;
      }
    }
  }

  // (2) passed!
}

TEST(adjacency_map2d, conforming) {
  // with footer adjacency map
  run_test_conforming("mortar/test_meshes/conforming_squarering/database.bin");
  // without footer
  run_test_conforming(
      "mortar/test_meshes/conforming_squarering/square_ring.bin");
}
