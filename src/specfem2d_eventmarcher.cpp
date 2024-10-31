
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "kernels/kernels.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"

#define _EVENT_MARCHER_DUMPS_
#define _stepwise_simfield_dump_ std::string("dump/simfield")
#define _index_change_dump_ std::string("dump/indexchange")

#include "_util/dump_simfield.hpp"
#include "_util/edge_storages.cpp"
#include "_util/rewrite_simfield.hpp"
#include "event_marching/event_marcher.hpp"
#include "event_marching/timescheme_wrapper.hpp"

#include <cmath>
#include <iostream>
#include <string>

#define _DUMP_INTERVAL_ 5

void execute(specfem::MPI::MPI *mpi) {
  // TODO sources / receivers
  // https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html

  std::vector<specfem::adjacency_graph::adjacency_pointer> edge_removals;
  auto params =
      _util::demo_assembly::simulation_params().dt(1e-2).tmax(5).use_demo_mesh(
          edge_removals);
  specfem::compute::assembly assembly = params.build_assembly();

#ifdef _EVENT_MARCHER_DUMPS_
  _util::init_dirs(_stepwise_simfield_dump_);
  _util::init_dirs(_index_change_dump_);

  _util::dump_simfield(_index_change_dump_ + "/prior_remap.dat",
                       assembly.fields.forward, assembly.mesh.points);
#endif

  // convert to compute indices
  for (int i = 0; i < edge_removals.size(); i++) {
    edge_removals[i].elem =
        assembly.mesh.mapping.mesh_to_compute(edge_removals[i].elem);
  }
  remap_with_disconts(assembly, params, edge_removals);

#ifdef _EVENT_MARCHER_DUMPS_
  _util::dump_simfield(_index_change_dump_ + "/post_remap.dat",
                       assembly.fields.forward, assembly.mesh.points);
#endif

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  auto kernels = specfem::kernels::kernels<
      specfem::wavefield::type::forward, specfem::dimension::type::dim2,
      specfem::enums::element::quadrature::static_quadrature_points<5> >(
      params.get_dt(), assembly, qp5);

  auto timescheme =
      specfem::time_scheme::newmark<specfem::simulation::type::forward>(
          params.get_numsteps(), 1, params.get_dt(), params.get_t0());
  timescheme.link_assembly(assembly);

  auto event_system = specfem::event_marching::event_system();

  auto timescheme_wrapper =
      specfem::event_marching::timescheme_wrapper(timescheme);

  constexpr auto acoustic = specfem::element::medium_tag::acoustic;
  constexpr auto elastic = specfem::element::medium_tag::elastic;
  timescheme_wrapper.set_forward_predictor_event(acoustic, 0);
  timescheme_wrapper.set_forward_predictor_event(elastic, 0.01);
  timescheme_wrapper.set_wavefield_update_event<acoustic>(kernels, 1);
  timescheme_wrapper.set_forward_corrector_event(acoustic, 2);
  timescheme_wrapper.set_wavefield_update_event<elastic>(kernels, 3);
  timescheme_wrapper.set_forward_corrector_event(elastic, 4);
  timescheme_wrapper
      .set_seismogram_update_event<specfem::wavefield::type::forward>(kernels,
                                                                      5);

  timescheme_wrapper.register_under_marcher(&event_system);

  specfem::event_marching::arbitrary_call_event reset_timer(
      [&]() {
        if (timescheme_wrapper.get_istep() < timescheme.get_max_timestep())
          event_system.set_current_precedence(
              specfem::event_marching::PRECEDENCE_BEFORE_INIT);
        return 0;
      },
      1);
  event_system.register_event(&reset_timer);
  specfem::event_marching::arbitrary_call_event output_fields(
      [&]() {
        int istep = timescheme_wrapper.get_istep();
        if (istep % _DUMP_INTERVAL_ == 0) {
          _util::dump_simfield_per_step(istep, _stepwise_simfield_dump_ + "/d",
                                        assembly.fields.forward,
                                        assembly.mesh.points);
        }
        return 0;
      },
      -0.1);
#ifdef _EVENT_MARCHER_DUMPS_
  event_system.register_event(&output_fields);
#endif

#define _IC_SIG 0.1
#define _IC_CENTER_X 0.3
#define _IC_CENTER_Z 0.6
  // initial condition
  set_field_disp<acoustic>(
      assembly.fields.forward, assembly.mesh, [](type_real x, type_real z) {
        x -= _IC_CENTER_X;
        z -= _IC_CENTER_Z;
        return (type_real)exp(-(x * x + z * z) / (2.0 * _IC_SIG * _IC_SIG));
      });
#undef _IC_SIG
#undef _IC_CENTER_X
#undef _IC_CENTER_Z

  // just populate dg_edges with edge_removals.
  auto edge_from_id = [&](const int8_t id) {
    switch (id) {
    case 0:
      return specfem::enums::edge::type::RIGHT;
    case 1:
      return specfem::enums::edge::type::TOP;
    case 2:
      return specfem::enums::edge::type::LEFT;
    case 3:
      return specfem::enums::edge::type::BOTTOM;
    default:
      return specfem::enums::edge::type::NONE;
    }
  };
  auto point_from_id = [&](int &ix, int &iz, const int8_t id, const int iedge) {
    switch (id) {
    case specfem::enums::edge::type::RIGHT:
      ix = qp5.NGLL - 1;
      iz = iedge;
      return;
    case specfem::enums::edge::type::TOP:
      ix = iedge;
      iz = qp5.NGLL - 1;
      return;
    case specfem::enums::edge::type::LEFT:
      ix = 0;
      iz = iedge;
      return;
    case specfem::enums::edge::type::BOTTOM:
      ix = iedge;
      iz = 0;
      return;
    default:
      return;
    }
  };
  std::vector<_util::edge_manager::edge> dg_edges(edge_removals.size());
  for (int i = 0; i < edge_removals.size(); i++) {
    dg_edges[i].id = edge_removals[i].elem;
    dg_edges[i].bdry = edge_from_id(edge_removals[i].side);
  }

#define EDGEIND_NX 0
#define EDGEIND_NZ 1
#define EDGEIND_DET 2
#define EDGEIND_DS 3
#define EDGEIND_FIELD 4
#define EDGEIND_FIELDNDERIV 6

#define data_capacity 10
#define edge_capacity 5
  _util::edge_manager::edge_storage<edge_capacity, data_capacity>
      dg_edge_storage(dg_edges);
  dg_edge_storage.foreach_edge_on_host(
      [&](_util::edge_manager::edge_data<edge_capacity, data_capacity> &e) {
        using PointPartialDerivativesType =
            specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                                true, false>;
        e.ngll = qp5.NGLL;
        int ix, iz;
        int ispec = e.parent.id;
        for (int i = 0; i < qp5.NGLL; i++) {
          point_from_id(ix, iz, e.parent.bdry, i);
          e.x[i] = assembly.mesh.points.coord(0, ispec, iz, ix);
          e.z[i] = assembly.mesh.points.coord(1, ispec, iz, ix);
          specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz,
                                                                      ix);

          PointPartialDerivativesType ppd;
          specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                         ppd);
          // type_real dvdxi = assembly.mesh.quadrature.h_hprime(ix,ix);
          // type_real dvdga = assembly.mesh.quadrature.h_hprime(iz,iz);
          // type_real dvdx = dvdxi * ppd.xix +
          //                  dvdga * ppd.gammax;
          // type_real dvdz = dvdxi * ppd.xiz +
          //                  dvdga * ppd.gammaz;
          type_real det = 1 / fabs(ppd.xix * ppd.gammaz - ppd.xiz * ppd.gammax);
          type_real nx, nz;
          switch (e.parent.bdry) {
          case specfem::enums::edge::type::RIGHT:
            nx = ppd.xix;
            nz = ppd.xiz;
            break;
          case specfem::enums::edge::type::TOP:
            nx = ppd.gammax;
            nx = ppd.gammaz;
            break;
          case specfem::enums::edge::type::LEFT:
            nx = -ppd.xix;
            nz = -ppd.xiz;
            break;
          case specfem::enums::edge::type::BOTTOM:
            nx = -ppd.gammax;
            nx = -ppd.gammaz;
            break;
          }
          e.data[i][EDGEIND_NX] = nx;
          e.data[i][EDGEIND_NZ] = nz;
          e.data[i][EDGEIND_DET] = det;
          e.data[i][EDGEIND_DS] = sqrt(nx * nx + nz * nz) * det; // dS
          // e.data[i][4] = (dvdx*nx + dvdz*nz)*det; //dv/dn dS
        }
      });
  specfem::event_marching::arbitrary_call_event store_boundaryvals(
      [&]() {
        _util::edge_manager::edge_storage<edge_capacity, data_capacity>
            dg_edge_storage(dg_edges);
        dg_edge_storage.foreach_edge_on_host(
            [&](_util::edge_manager::edge_data<edge_capacity, data_capacity>
                    &e) {
              using PointPartialDerivativesType =
                  specfem::point::partial_derivatives<
                      specfem::dimension::type::dim2, true, false>;
              using PointAcoustic =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        false, false, true, false, false>;
              using PointElastic =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::elastic,
                                        false, false, true, false, false>;
              int ix, iz;
              int ispec = e.parent.id;
              for (int i = 0; i < qp5.NGLL; i++) {
                point_from_id(ix, iz, e.parent.bdry, i);
                specfem::point::index<specfem::dimension::type::dim2> index(
                    ispec, iz, ix);

                PointPartialDerivativesType ppd;
                specfem::compute::load_on_host(
                    index, assembly.partial_derivatives, ppd);
                int ncomp;
                std::vector<type_real> dfdxi;
                std::vector<type_real> dfdga;
                specfem::element::medium_tag medium =
                    assembly.properties.h_element_types(ispec);
                if (medium == specfem::element::medium_tag::acoustic) {
                  constexpr int medium_ID =
                      static_cast<int>(specfem::element::medium_tag::acoustic);
                  ncomp = 1;
                  dfdxi = std::vector<type_real>(ncomp, 0);
                  dfdga = std::vector<type_real>(ncomp, 0);
                  int icomp = 0;
                  for (int k = 0; k < qp5.NGLL; k++) {
                    dfdxi[icomp] +=
                        assembly.mesh.quadrature.h_hprime(ix, k) *
                        assembly.fields.forward.acoustic.h_field(
                            assembly.fields.forward.h_assembly_index_mapping(
                                assembly.fields.forward.h_index_mapping(ispec,
                                                                        iz, k),
                                medium_ID),
                            icomp);
                    dfdga[icomp] +=
                        assembly.mesh.quadrature.h_hprime(iz, k) *
                        assembly.fields.forward.acoustic.h_field(
                            assembly.fields.forward.h_assembly_index_mapping(
                                assembly.fields.forward.h_index_mapping(ispec,
                                                                        k, ix),
                                medium_ID),
                            icomp);
                  }
                  e.data[i][EDGEIND_FIELD + 0] =
                      assembly.fields.forward.acoustic.h_field(
                          assembly.fields.forward.h_assembly_index_mapping(
                              assembly.fields.forward.h_index_mapping(ispec, iz,
                                                                      ix),
                              0),
                          0);
                } else {
                  throw std::runtime_error(
                      "Flux edge-storage: medium not supported.");
                }
                std::vector<type_real> dfdx(ncomp, 0);
                std::vector<type_real> dfdz(ncomp, 0);
                for (int icomp = 0; icomp < ncomp; icomp++) {
                  dfdx[icomp] =
                      dfdxi[icomp] * ppd.xix + dfdga[icomp] * ppd.gammax;
                  dfdz[icomp] =
                      dfdxi[icomp] * ppd.xiz + dfdga[icomp] * ppd.gammaz;
                }

                type_real nx = e.data[i][EDGEIND_NX];
                type_real nz = e.data[i][EDGEIND_NZ];
                type_real det = e.data[i][EDGEIND_DET];
                for (int icomp = 0; icomp < ncomp; icomp++) {
                  e.data[i][EDGEIND_FIELDNDERIV + icomp] =
                      (dfdx[icomp] * nx + dfdz[icomp] * nz) * det;
                }
              }
            });
        return 0;
      },
      0.9);
  timescheme_wrapper.time_stepper.register_event(&store_boundaryvals);

  dg_edge_storage.foreach_intersection_on_host(
      [&](_util::edge_manager::edge_intersection &intersect,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &b) {
        //
      });

  // init kernels needs to happen as of time_marcher
  kernels.initialize(timescheme.get_timestep());

  // ================================RUNTIME
  // BEGIN=================================
  // event_system.run();
#undef edge_capacity
#undef data_capacity
}

int main(int argc, char **argv) {

  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  { execute(mpi); }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
