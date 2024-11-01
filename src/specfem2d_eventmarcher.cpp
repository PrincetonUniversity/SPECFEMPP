
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "kernels/kernels.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"

#define _RELAX_PARAM_COEF_ 40

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
      _util::demo_assembly::simulation_params().dt(1e-3).tmax(5).use_demo_mesh(
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

#define _IC_SIG 0.05
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
#define EDGEIND_SPEEDPARAM 8
#define EDGEIND_SHAPENDERIV 9

#define data_capacity 20
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
          // type_real dvdxi = assembly.mesh.quadratures.h_hprime(ix,ix);
          // type_real dvdga = assembly.mesh.quadratures.h_hprime(iz,iz);
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
            nz = ppd.gammaz;
            break;
          case specfem::enums::edge::type::LEFT:
            nx = -ppd.xix;
            nz = -ppd.xiz;
            break;
          case specfem::enums::edge::type::BOTTOM:
            nx = -ppd.gammax;
            nz = -ppd.gammaz;
            break;
          }
          e.data[EDGEIND_NX][i] = nx;
          e.data[EDGEIND_NZ][i] = nz;
          e.data[EDGEIND_DET][i] = det;
          e.data[EDGEIND_DS][i] = sqrt(nx * nx + nz * nz) * det; // dS
          for (int ishape = 0; ishape < e.ngll; ishape++) {
            int ixshape, izshape;
            point_from_id(ixshape, izshape, e.parent.bdry, ishape);
            // v = L_{ixshape}(x) L_{izshape}(z);
            // hprime(i,j) = L_j'(t_i)
            type_real dvdxi =
                assembly.mesh.quadratures.gll.h_hprime(ix, ixshape) *
                (iz == izshape);
            type_real dvdga =
                assembly.mesh.quadratures.gll.h_hprime(iz, izshape) *
                (ix == ixshape);
            type_real dvdx = dvdxi * ppd.xix + dvdga * ppd.gammax;
            type_real dvdz = dvdxi * ppd.xiz + dvdga * ppd.gammaz;
            e.data[EDGEIND_SHAPENDERIV + ishape][i] =
                (dvdx * nx + dvdz * nz) * det;
          }
        }
      });
  auto spec_charlen2 = [&](int ispec) {
    type_real lx1 =
        assembly.mesh.points.coord(0, ispec, 0, 0) -
        assembly.mesh.points.coord(0, ispec, qp5.NGLL - 1, qp5.NGLL - 1);
    type_real lz1 =
        assembly.mesh.points.coord(1, ispec, 0, 0) -
        assembly.mesh.points.coord(1, ispec, qp5.NGLL - 1, qp5.NGLL - 1);
    type_real lx2 = assembly.mesh.points.coord(0, ispec, qp5.NGLL - 1, 0) -
                    assembly.mesh.points.coord(0, ispec, 0, qp5.NGLL - 1);
    type_real lz2 = assembly.mesh.points.coord(1, ispec, qp5.NGLL - 1, 0) -
                    assembly.mesh.points.coord(1, ispec, 0, qp5.NGLL - 1);
    return std::max(lx1 * lx1 + lz1 * lz1, lx2 * lx2 + lz2 * lz2);
  };
  dg_edge_storage.foreach_intersection_on_host(
      [&](_util::edge_manager::edge_intersection<edge_capacity> &intersect,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &b) {
        int a_ispec = a.parent.id;
        int b_ispec = b.parent.id;
        specfem::element::medium_tag a_medium =
            assembly.properties.h_element_types(a_ispec);
        specfem::element::medium_tag b_medium =
            assembly.properties.h_element_types(b_ispec);
        if (a_medium == specfem::element::medium_tag::acoustic &&
            b_medium == specfem::element::medium_tag::acoustic) {
          type_real rho_inv_max = 0;
          for (int iz = 0; iz < qp5.NGLL; iz++) {
            for (int ix = 0; ix < qp5.NGLL; ix++) {
              rho_inv_max =
                  std::max(rho_inv_max,
                           assembly.properties.acoustic_isotropic.h_rho_inverse(
                               a_ispec, iz, ix));
              rho_inv_max =
                  std::max(rho_inv_max,
                           assembly.properties.acoustic_isotropic.h_rho_inverse(
                               b_ispec, iz, ix));
            }
          }
          intersect.relax_param =
              _RELAX_PARAM_COEF_ * rho_inv_max /
              sqrt(std::max(spec_charlen2(a_ispec), spec_charlen2(b_ispec)));

        } else {
          throw std::runtime_error(
              "Mortar Flux: medium combination not supported.");
        }
      });

  specfem::event_marching::arbitrary_call_event store_boundaryvals(
      [&]() {
        dg_edge_storage.foreach_edge_on_host([&](_util::edge_manager::edge_data<
                                                 edge_capacity, data_capacity>
                                                     &e) {
          using PointPartialDerivativesType =
              specfem::point::partial_derivatives<
                  specfem::dimension::type::dim2, true, false>;
          using PointAcoustic =
              specfem::point::field<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::acoustic,
                                    true, false, false, false, false>;
          using PointElastic =
              specfem::point::field<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::elastic, true,
                                    false, false, false, false>;
          int ix, iz;
          int ispec = e.parent.id;
          for (int i = 0; i < qp5.NGLL; i++) {
            point_from_id(ix, iz, e.parent.bdry, i);
            specfem::point::index<specfem::dimension::type::dim2> index(ispec,
                                                                        iz, ix);

            PointPartialDerivativesType ppd;
            specfem::compute::load_on_host(index, assembly.partial_derivatives,
                                           ppd);
            int ncomp;
            std::vector<type_real> dfdxi;
            std::vector<type_real> dfdga;
            specfem::element::medium_tag medium =
                assembly.properties.h_element_types(ispec);
            if (medium == specfem::element::medium_tag::acoustic) {
              PointAcoustic disp;
              constexpr int medium_ID =
                  static_cast<int>(specfem::element::medium_tag::acoustic);
              ncomp = 1;
              dfdxi = std::vector<type_real>(ncomp, 0);
              dfdga = std::vector<type_real>(ncomp, 0);
              int icomp = 0;
              specfem::compute::load_on_host(index, assembly.fields.forward,
                                             disp);
              e.data[EDGEIND_FIELD][i] = disp.displacement(0);
              for (int k = 0; k < qp5.NGLL; k++) {
                specfem::point::index<specfem::dimension::type::dim2> index_(
                    ispec, iz, k);
                specfem::compute::load_on_host(index_, assembly.fields.forward,
                                               disp);
                dfdxi[icomp] += assembly.mesh.quadratures.gll.h_hprime(ix, k) *
                                disp.displacement(0);
                index_ = specfem::point::index<specfem::dimension::type::dim2>(
                    ispec, iz, k);
                specfem::compute::load_on_host(index_, assembly.fields.forward,
                                               disp);
                dfdga[icomp] += assembly.mesh.quadratures.gll.h_hprime(iz, k) *
                                disp.displacement(0);
              }
              e.data[EDGEIND_SPEEDPARAM][i] =
                  assembly.properties.acoustic_isotropic.h_rho_inverse(ispec,
                                                                       iz, ix);
            } else {
              throw std::runtime_error(
                  "Flux edge-storage: medium not supported.");
            }
            std::vector<type_real> dfdx(ncomp, 0);
            std::vector<type_real> dfdz(ncomp, 0);
            for (int icomp = 0; icomp < ncomp; icomp++) {
              dfdx[icomp] = dfdxi[icomp] * ppd.xix + dfdga[icomp] * ppd.gammax;
              dfdz[icomp] = dfdxi[icomp] * ppd.xiz + dfdga[icomp] * ppd.gammaz;
            }

            type_real nx = e.data[EDGEIND_NX][i];
            type_real nz = e.data[EDGEIND_NZ][i];
            type_real det = e.data[EDGEIND_DET][i];
            for (int icomp = 0; icomp < ncomp; icomp++) {
              e.data[EDGEIND_FIELDNDERIV + icomp][i] =
                  (dfdx[icomp] * nx + dfdz[icomp] * nz) * det;
            }
          }
        });
        return 0;
      },
      0.9);
  timescheme_wrapper.time_stepper.register_event(&store_boundaryvals);

  specfem::event_marching::arbitrary_call_event mortar_flux(
      [&]() {
        dg_edge_storage.foreach_intersection_on_host(
            [&](_util::edge_manager::edge_intersection<edge_capacity>
                    &intersect,
                _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
                _util::edge_manager::edge_data<edge_capacity, data_capacity>
                    &b) {
              using PointAcoustic =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        true, false, false, false, false>;
              using PointElastic =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::elastic,
                                        true, false, false, false, false>;
              using PointAcousticAccel =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::acoustic,
                                        false, false, true, false, false>;
              using PointElasticAccel =
                  specfem::point::field<specfem::dimension::type::dim2,
                                        specfem::element::medium_tag::elastic,
                                        false, false, true, false, false>;
              _util::edge_manager::quadrature_rule gll =
                  _util::edge_manager::gen_GLL(intersect.ngll);
              int a_ispec = a.parent.id;
              int b_ispec = b.parent.id;
              specfem::element::medium_tag a_medium =
                  assembly.properties.h_element_types(a_ispec);
              specfem::element::medium_tag b_medium =
                  assembly.properties.h_element_types(b_ispec);
              if (a.data[EDGEIND_FIELD][2] > 0.2) {
                intersect.a_to_mortar(1, a.data[EDGEIND_FIELDNDERIV]);
              }
              if (a_medium == specfem::element::medium_tag::acoustic &&
                  b_medium == specfem::element::medium_tag::acoustic) {
                for (int a_ishape = 0; a_ishape < a.ngll; a_ishape++) {
                  // flux += (
                  //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                  //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                  //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                  // )
                  type_real flux = 0;
                  for (int iquad = 0; iquad < gll.nquad; iquad++) {
                    type_real c = intersect.a_to_mortar(
                        iquad, a.data[EDGEIND_SPEEDPARAM]);
                    type_real u_jmp =
                        intersect.a_to_mortar(iquad, a.data[EDGEIND_FIELD]) -
                        intersect.b_to_mortar(iquad, b.data[EDGEIND_FIELD]);
                    type_real cdu_avg =
                        0.5 * (c * intersect.a_to_mortar(
                                       iquad, a.data[EDGEIND_FIELDNDERIV]) -
                               intersect.b_to_mortar(
                                   iquad, b.data[EDGEIND_SPEEDPARAM]) *
                                   intersect.b_to_mortar(
                                       iquad, b.data[EDGEIND_FIELDNDERIV]));
                    flux += gll.w[iquad] *
                            (0.5 *
                                 intersect.a_to_mortar(
                                     iquad,
                                     a.data[EDGEIND_SHAPENDERIV + a_ishape]) *
                                 c * u_jmp +
                             intersect.a_to_mortar(iquad, a.data[EDGEIND_DS]) *
                                 intersect.a_mortar_trans[iquad][a_ishape] *
                                 (cdu_avg - intersect.relax_param * u_jmp));
                  }
                  PointAcousticAccel accel;
                  specfem::point::index<specfem::dimension::type::dim2> index(
                      a_ispec, 0, 0);
                  point_from_id(index.ix, index.iz, a.parent.bdry, a_ishape);
                  accel.acceleration = flux;
                  specfem::compute::atomic_add_on_device(
                      index, accel, assembly.fields.forward);
                }
                for (int b_ishape = 0; b_ishape < b.ngll; b_ishape++) {
                  // flux += (
                  //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                  //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                  //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                  // )
                  type_real flux = 0;
                  for (int iquad = 0; iquad < gll.nquad; iquad++) {
                    type_real c = intersect.b_to_mortar(
                        iquad, b.data[EDGEIND_SPEEDPARAM]);
                    type_real u_jmp =
                        intersect.b_to_mortar(iquad, b.data[EDGEIND_FIELD]) -
                        intersect.a_to_mortar(iquad, a.data[EDGEIND_FIELD]);
                    type_real cdu_avg =
                        0.5 * (c * intersect.b_to_mortar(
                                       iquad, b.data[EDGEIND_FIELDNDERIV]) -
                               intersect.a_to_mortar(
                                   iquad, a.data[EDGEIND_SPEEDPARAM]) *
                                   intersect.a_to_mortar(
                                       iquad, a.data[EDGEIND_FIELDNDERIV]));
                    flux += gll.w[iquad] *
                            (0.5 *
                                 intersect.b_to_mortar(
                                     iquad,
                                     b.data[EDGEIND_SHAPENDERIV + b_ishape]) *
                                 c * u_jmp +
                             intersect.b_to_mortar(iquad, b.data[EDGEIND_DS]) *
                                 intersect.b_mortar_trans[iquad][b_ishape] *
                                 (cdu_avg - intersect.relax_param * u_jmp));
                  }
                  PointAcousticAccel accel;
                  specfem::point::index<specfem::dimension::type::dim2> index(
                      b_ispec, 0, 0);
                  point_from_id(index.ix, index.iz, b.parent.bdry, b_ishape);
                  accel.acceleration = flux;
                  specfem::compute::atomic_add_on_device(
                      index, accel, assembly.fields.forward);
                }

              } else {
                throw std::runtime_error(
                    "Mortar Flux: medium combination not supported.");
              }
            });
        return 0;
      },
      0.91);
  timescheme_wrapper.time_stepper.register_event(&mortar_flux);

  // init kernels needs to happen as of time_marcher
  kernels.initialize(timescheme.get_timestep());

  // ================================RUNTIME
  // BEGIN=================================
  event_system.run();
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
