#include <gtest/gtest.h>
#include <string>

#include "../../MPI_environment.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/interface.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

TEST(adjacency_map2d, conforming) {
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto mesh = specfem::io::read_2d_mesh(
      "mortar/test_meshes/conforming_squarering/square_ring.bin",
      specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, mpi);

  mpi->cout("Mesh read. Forming adjacency map.");
  specfem::mesh::adjacency_map::adjacency_map<specfem::dimension::type::dim2>
      adjacencies(mesh);
  std::ostringstream msg;
  msg << "Adjacency map formed (nspec = " << adjacencies.nspec
      << "). Comparing results...";
  mpi->cout(msg.str());

  // print out adjacencies
  msg = std::ostringstream();
#define COLWIDTH (7)
#define NUMCOLS (5)
#define NUM_DIGITS(st) (st < 10 ? 1 : (st < 100 ? 2 : (st < 1000 ? 3 : 4)))
  char entry[COLWIDTH];
  const auto print_entry = [&](bool terminate = false){
    int stsize;
    for(stsize = 0; stsize < COLWIDTH && entry[stsize] != '\0'; stsize++){}
    int padsize = COLWIDTH - stsize;
    for(int i = padsize/2; i > 0; i--){
      msg << ' ';
    }
    msg.write(entry, stsize);
    for(int i = padsize - padsize/2; i > 0; i--){
      msg << ' ';
    }
    if(terminate){
      msg << '\n';
    }else{
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
          entry, "%d%c",
          ispec_adj,
          type_adj == specfem::enums::edge::type::TOP
              ? 'T'
              : (type_adj == specfem::enums::edge::type::BOTTOM
                     ? 'B'
                     : (type_adj == specfem::enums::edge::type::LEFT ? 'L'
                                                                     : 'R')));
    }else if(adjacencies.has_boundary(ispec,type)){
      std::sprintf(entry, "(bdry)");
    }else{
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
  for (int i = 0; i < adjacencies.nspec; i++) {
    std::sprintf(entry, "%d", i);
    print_entry();
    set_entry_from_adj(i,specfem::enums::edge::type::RIGHT);
    print_entry();
    set_entry_from_adj(i,specfem::enums::edge::type::TOP);
    print_entry();
    set_entry_from_adj(i,specfem::enums::edge::type::LEFT);
    print_entry();
    set_entry_from_adj(i,specfem::enums::edge::type::BOTTOM);
    print_entry(true);
  }
  mpi->cout(msg.str());
}