
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "timescheme/newmark.hpp"
#include "solver/time_marching.hpp"
#include "kernels/kernels.hpp"

#include "event_marching/event_marcher.hpp"
#include "event_marching/timemarching_wrapper.hpp"
#include "_util/rewrite_simfield.hpp"


#include <iostream>


//#define _EVENT_MARCHER_VERBOSE_

void execute(specfem::MPI::MPI *mpi){
  // TODO sources / receivers
// https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html


  std::vector<specfem::adjacency_graph::adjacency_pointer> edge_removals;
  auto params = _util::demo_assembly::simulation_params()
        .dt(1e-3).tmax(5).use_demo_mesh(edge_removals);
  specfem::compute::assembly assembly = params.build_assembly();

#ifdef _EVENT_MARCHER_VERBOSE_
  std::cout << "prior_inds = ";
  _util::print_index_mappings(assembly.fields.forward);
  std::cout << "\n";
  std::cout << "prior_pts = ";
  _util::print_point_locations<2>(assembly);
  std::cout << "\n";
#endif

  remap_with_disconts(assembly.fields.forward,assembly.mesh,assembly.properties,edge_removals);

#ifdef _EVENT_MARCHER_VERBOSE_
  std::cout << "post_inds = ";
  _util::print_index_mappings(assembly.fields.forward);
  std::cout << "\n";
  // std::cout << "post_pts = ";
  // _util::print_point_locations<2>(assembly);
  // std::cout << "\n";
  std::cout << "\n\n\n\n\n\n\n\n\n" << std::flush;
#endif


  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  auto kernels = specfem::kernels::kernels<
          specfem::wavefield::type::forward,
          specfem::dimension::type::dim2,
          specfem::enums::element::quadrature::static_quadrature_points<5>
          >(params.get_dt(),assembly,qp5);


  auto timescheme = std::make_shared<specfem::time_scheme::newmark<specfem::simulation::type::forward>>(
    params.get_numsteps(), 1, params.get_dt(), params.get_t0()
  );
  timescheme->link_assembly(assembly);
  
  auto marcher = specfem::event_marching::event_marcher();
  auto time_stepping = specfem::solver::time_marching<specfem::simulation::type::forward,
                                               specfem::dimension::type::dim2,
                                               specfem::enums::element::quadrature::static_quadrature_points<5>>
        (kernels,timescheme);
  

  auto timemarching_wrapper = specfem::event_marching::timemarching_wrapper<
          specfem::simulation::type::forward, specfem::dimension::type::dim2,
          specfem::enums::element::quadrature::static_quadrature_points<5>>(time_stepping);

  //TODO mechanism to register wrapper into marcher (populate marcher events).
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  timemarching_wrapper.set_forward_predictor_precedence(acoustic,0);
  timemarching_wrapper.set_forward_predictor_precedence(elastic,0.01);
  timemarching_wrapper.set_wavefield_update_precedence(acoustic,1);
  timemarching_wrapper.set_forward_corrector_precedence(acoustic,2);
  timemarching_wrapper.set_wavefield_update_precedence(elastic,3);
  timemarching_wrapper.set_forward_corrector_precedence(elastic,4);
  timemarching_wrapper.set_seismogram_update_precedence(5);

  timemarching_wrapper.load_into_marcher_main_events(marcher);
  time_stepping.init_kernels();
  marcher.run();

  
  // auto solver = specfem::solver::time_marching<specfem::simulation::type::forward,
  //                                              specfem::dimension::type::dim2,
  //                                              specfem::enums::element::quadrature::static_quadrature_points<5>>
  //       (kernels,timescheme);
  // solver.run();
}


int main(int argc, char **argv) {

  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    execute(mpi);
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}