#ifndef __UTIL_REWRITE_SIMFIELD_HPP_
#define __UTIL_REWRITE_SIMFIELD_HPP_

#include "compute/fields/simulation_field.hpp"
#include "adjacency_graph/adjacency_graph.hpp"

template <specfem::wavefield::type WavefieldType>
void remap_with_disconts(specfem::compute::simulation_field<WavefieldType> & field,
    const specfem::compute::mesh &mesh,
    const specfem::compute::properties &properties,
    const std::vector<specfem::adjacency_graph::adjacency_pointer>& removals){
  using ViewType =
      Kokkos::View<int ***, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>;
  
  auto graph = specfem::adjacency_graph::from_index_mapping(field.h_index_mapping);
  specfem::adjacency_graph::adjacency_pointer null_adj;


#ifdef _EVENT_MARCHER_VERBOSE_
  std::cout << "adjs_prior = [";
  for(int i = 0; i < graph.get_size(); i++){
    std::cout << "[";
    for(int j = 0; j < 4; j++){
      auto adj = graph.get_adjacency(i,j);
      std::cout << "[" << adj.elem << "," << (int)adj.side << "],";
    }
    std::cout << "],";
  }
  std::cout << "]\n";
#endif

  //remove edges
  for(int i = 0; i < removals.size(); i++){
    graph.form_adjacency(removals[i],null_adj);
  }

#ifdef _EVENT_MARCHER_VERBOSE_
  std::cout << "adjs_post = [";
  for(int i = 0; i < graph.get_size(); i++){
    std::cout << "[";
    for(int j = 0; j < 4; j++){
      auto adj = graph.get_adjacency(i,j);
      std::cout << "[" << adj.elem << "," << (int)adj.side << "],";
    }
    std::cout << "],";
  }
  std::cout << "]\n";
#endif

  //rebuild index_mapping
  int nglob;
  ViewType index_mapping = specfem::adjacency_graph::to_index_mapping(graph, field.ngllz, field.ngllx, &nglob);
  ViewType::HostMirror h_index_mapping = ViewType::HostMirror(Kokkos::create_mirror_view(index_mapping));


  field.nglob = nglob;
  field.index_mapping = index_mapping;
  field.h_index_mapping = h_index_mapping;

  field.set_assembly_index_mappings(mesh,properties);
}

#endif