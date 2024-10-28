
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "timescheme/newmark.hpp"
#include "solver/time_marching.hpp"
#include "kernels/kernels.hpp"

#include "event_marching/event_marcher.hpp"
#include "event_marching/timescheme_wrapper.hpp"
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


  auto timescheme = specfem::time_scheme::newmark<specfem::simulation::type::forward>(
    params.get_numsteps(), 1, params.get_dt(), params.get_t0()
  );
  timescheme.link_assembly(assembly);
  
  auto event_system = specfem::event_marching::event_system();
  

  auto timescheme_wrapper = specfem::event_marching::timescheme_wrapper(timescheme);

  //TODO mechanism to register wrapper into marcher (populate marcher events).
  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  timescheme_wrapper.set_forward_predictor_event(acoustic,0);
  timescheme_wrapper.set_forward_predictor_event(elastic,0.01);
  timescheme_wrapper.set_wavefield_update_event<acoustic>(kernels,1);
  timescheme_wrapper.set_forward_corrector_event(acoustic,2);
  timescheme_wrapper.set_wavefield_update_event<elastic>(kernels,3);
  timescheme_wrapper.set_forward_corrector_event(elastic,4);
  timescheme_wrapper.set_seismogram_update_event<specfem::wavefield::type::forward>(kernels,5);

  timescheme_wrapper.register_under_marcher(&event_system);


  specfem::event_marching::arbitrary_call_event reset_timer([&]() {event_system.set_current_precedence(0); return 0;},1);
  event_system.register_event(&reset_timer);

  kernels.initialize(timescheme.get_timestep());
  event_system.run();

  
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