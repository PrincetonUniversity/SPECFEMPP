#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <tuple>
#include <utility>

#include "../../MPI_environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/interface.hpp"
#include "macros.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "mortar/fixture/mortar_fixtures.hpp"

#define TEST_ASSEMBLY_MAPPING_DEFAULT_NGLL (5)
// TODO: we may want to generalize this test -- it may be useful for a wider
// range
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

        if (0 > ind || ind >= nglob) {
          FAIL() << "Index mapping maps to an out-of-bounds index! (" << ind
                 << ")";
        }
        ind_to_nodes[ind].push_back(std::make_tuple(ispec, iz, ix));
      }
    }
  }

  // is each index used?
  for (int iglob = 0; iglob < nglob; iglob++) {
    if (ind_to_nodes[iglob].empty()) {
      FAIL() << "Index mapping not surjective. (" << iglob
             << " has no preimage.)";
    }
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
        [&]() {
          FAIL() << "Test internally incorrect: local_to_bdry lambda should "
                 << "not have been called with internal node indices (ix = "
                 << ix << ", iz = " << iz << ").";
        }();
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
      if (nodeset.size() != 1) {
        FAIL() << "Internal node (ispec = " << ispec << ", ix = " << ix
               << ", iz = " << iz << ") should not share global index ("
               << iglob << "), but shares with " << nodeset.size()
               << " other elements.";
      }
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

void run_test_conforming(test_configuration::mesh &mesh_config) {
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  auto mesh = specfem::io::read_2d_mesh(
      mesh_config.database, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, mpi);

  mpi->cout("Mesh read. Forming adjacency map.");
  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
      &adjacencies = mesh.adjacency_map;
  if (!adjacencies.was_initialized()) {
    throw std::runtime_error("Adjacency map not built.");
  }

  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
      provenance_adjmap = mesh_config.reference_adjacency_map();

  std::ostringstream failmsg;
  bool fail = false;
  // compare conforming adjacency maps
  for (int ispec = 0; ispec < mesh.nspec; ispec++) {
    for (auto edge : {
             specfem::enums::edge::type::RIGHT,
             specfem::enums::edge::type::TOP,
             specfem::enums::edge::type::LEFT,
             specfem::enums::edge::type::BOTTOM,
         }) {
      bool true_has_adj =
          provenance_adjmap.has_conforming_adjacency(ispec, edge);
      if (true_has_adj != adjacencies.has_conforming_adjacency(ispec, edge)) {
        fail = true;
        if (true_has_adj) {
          failmsg << " - Did not find an adjacency along edge "
                  << adjacencies.edge_to_string(ispec, edge) << "\n"
                  << "     expected: "
                  << std::apply(adjacencies.edge_to_string,
                                provenance_adjmap.get_conforming_adjacency(
                                    ispec, edge))
                  << "\n";
        } else {
          failmsg << " - Found an adjacency along edge "
                  << adjacencies.edge_to_string(ispec, edge) << "\n"
                  << "     (should not be linked)\n";
        }
        break;
      }
      // agreement on if conforming adjacency exists. Do we have the right one?
      if (true_has_adj &&
          (adjacencies.get_conforming_adjacency(ispec, edge) !=
           provenance_adjmap.get_conforming_adjacency(ispec, edge))) {
        fail = true;
        failmsg << " - Incorrect adjacency map along edge "
                << adjacencies.edge_to_string(ispec, edge) << "\n"
                << "     expected: "
                << std::apply(
                       adjacencies.edge_to_string,
                       provenance_adjmap.get_conforming_adjacency(ispec, edge))
                << "\n        found: "
                << std::apply(adjacencies.edge_to_string,
                              adjacencies.get_conforming_adjacency(ispec, edge))
                << "\n";
        break;
      }
    }
  }
  if (fail) {
    FAIL() << "--------------------------------------------------\n"
           << "\033[0;31m[FAILED]\033[0m Test failed\n"
           << "               conforming\n"
           << failmsg.str()
           << "--------------------------------------------------\n"
           << std::endl;
  }

  test_assembly_mapping(adjacencies, mesh.nspec,
                        TEST_ASSEMBLY_MAPPING_DEFAULT_NGLL);
}

TEST_F(MESHES, conforming) {
  for (auto mesh : *this) {
    try {
      run_test_conforming(mesh);
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m " << mesh.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test: " << mesh.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
