
#include "_util/build_demo_assembly.hpp"
#include "compute/assembly/assembly.hpp"
#include "kernels/kernels.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"

// #define FORCE_INTO_CONTINUOUS
// #define USE_DEMO_MESH
// #define SET_INITIAL_CONDITION
#define _RELAX_PARAM_COEF_ACOUSTIC_ 40
#define _RELAX_PARAM_COEF_ELASTIC_ 40

#define _EVENT_MARCHER_DUMPS_
#define _stepwise_simfield_dump_ std::string("dump/simfield")
#define _index_change_dump_ std::string("dump/indexchange")
#define _stepwise_edge_dump_ std::string("dump/edgedata")
#define _PARAMETER_FILENAME_ std::string("specfem_config.yaml")

#include "_util/dump_simfield.hpp"
#include "_util/edge_storages.hpp"
#include "_util/rewrite_simfield.hpp"
#include "event_marching/event_marcher.hpp"
#include "event_marching/timescheme_wrapper.hpp"

#include <cmath>
#include <iostream>
#include <string>

#define _DUMP_INTERVAL_ 5
_util::demo_assembly::simulation_params
load_parameters(const std::string &parameter_file, specfem::MPI::MPI *mpi);

void execute(specfem::MPI::MPI *mpi) {
  // TODO sources / receivers
  // https://specfem2d-kokkos.readthedocs.io/en/adjoint-simulations/developer_documentation/tutorials/tutorial1/Chapter2/index.html

  std::vector<specfem::adjacency_graph::adjacency_pointer> edge_removals;
#ifdef USE_DEMO_MESH
#ifdef FORCE_INTO_CONTINUOUS
  auto params =
      _util::demo_assembly::simulation_params().dt(1e-3).tmax(5).use_demo_mesh(
          0,
          edge_removals); // construct-mode == 0 (no shifts)
  edge_removals.clear();
#else
  auto params =
      _util::demo_assembly::simulation_params().dt(1e-3).tmax(5).use_demo_mesh(
          1,
          edge_removals); // construct-mode == 1
#endif
#else
  auto params = load_parameters(_PARAMETER_FILENAME_, mpi);

#endif
  std::shared_ptr<specfem::compute::assembly> assembly = params.get_assembly();

#ifdef _EVENT_MARCHER_DUMPS_
  _util::init_dirs(_stepwise_simfield_dump_);
  _util::init_dirs(_index_change_dump_);
  _util::init_dirs(_stepwise_edge_dump_);

  _util::dump_simfield(_index_change_dump_ + "/prior_remap.dat",
                       assembly->fields.forward, assembly->mesh.points);
#endif

  // convert to compute indices
  for (int i = 0; i < edge_removals.size(); i++) {
    edge_removals[i].elem =
        assembly->mesh.mapping.mesh_to_compute(edge_removals[i].elem);
  }
#ifndef FORCE_INTO_CONTINUOUS
  remap_with_disconts(*assembly, params, edge_removals, true);
#endif

#ifdef _EVENT_MARCHER_DUMPS_
  _util::dump_simfield(_index_change_dump_ + "/post_remap.dat",
                       assembly->fields.forward, assembly->mesh.points);
#endif

  specfem::enums::element::quadrature::static_quadrature_points<5> qp5;
  auto kernels = specfem::kernels::kernels<
      specfem::wavefield::simulation_field::forward,
      specfem::dimension::type::dim2,
      specfem::enums::element::quadrature::static_quadrature_points<5> >(
      params.get_dt(), *assembly, qp5);

  auto timescheme =
      specfem::time_scheme::newmark<specfem::simulation::type::forward>(
          params.get_numsteps(), 1, params.get_dt(), params.get_t0());
  timescheme.link_assembly(*assembly);

  params.set_plotters_from_runtime_configuration();

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
  timescheme_wrapper.set_seismogram_update_event<
      specfem::wavefield::simulation_field::forward>(kernels, 5);
  timescheme_wrapper.set_plotter_update_event(params.get_plotters(), 5.1);

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
                                        assembly->fields.forward,
                                        assembly->mesh.points);
        }
        return 0;
      },
      -0.1);
#ifdef _EVENT_MARCHER_DUMPS_
  event_system.register_event(&output_fields);
#endif

#ifdef SET_INITIAL_CONDITION
#define _IC_SIG 0.05
#define _IC_CENTER_X 0.3
#define _IC_CENTER_Z 0.6
  // initial condition
  set_field_disp<acoustic>(
      assembly->fields.forward, assembly->mesh, [](type_real x, type_real z) {
        x -= _IC_CENTER_X;
        z -= _IC_CENTER_Z;
        return (type_real)exp(-(x * x + z * z) / (2.0 * _IC_SIG * _IC_SIG));
      });
#undef _IC_SIG
#undef _IC_CENTER_X
#undef _IC_CENTER_Z
#endif

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
#define EDGEIND_BDRY_TYPE 14

#define data_capacity 20
#define edge_capacity 5
#define INTERIND_FLUXTOTAL_A 0
#define INTERIND_FLUX1_A (qp5.NGLL)
#define INTERIND_FLUX2_A (qp5.NGLL * 2)
#define INTERIND_FLUX3_A (qp5.NGLL * 3)
#define INTERIND_FLUXTOTAL_B (qp5.NGLL * 4)
#define INTERIND_FLUX1_B (qp5.NGLL * 5)
#define INTERIND_FLUX2_B (qp5.NGLL * 6)
#define INTERIND_FLUX3_B (qp5.NGLL * 7)
#define INTERIND_UJMP (qp5.NGLL * 8)
#define INTERIND_DU_AVG (qp5.NGLL * 9)
#define INTERIND_ACCEL_INCLUDE_A (qp5.NGLL * 10)
#define INTERIND_ACCEL_INCLUDE_B (qp5.NGLL * 11)

#define intersect_data_capacity (qp5.NGLL * (12))
  _util::edge_manager::edge_storage<edge_capacity, data_capacity>
      dg_edge_storage(dg_edges);

  specfem::event_marching::arbitrary_call_event output_edges(
      [&]() {
        int istep = timescheme_wrapper.get_istep();
        if (istep % _DUMP_INTERVAL_ == 0) {
          _util::dump_edge_container(_stepwise_edge_dump_ + "/d" +
                                         std::to_string(istep) + ".dat",
                                     dg_edge_storage);
        }
        return 0;
      },
      -0.1);
#ifdef _EVENT_MARCHER_DUMPS_
  event_system.register_event(&output_edges);
#endif
  dg_edge_storage.foreach_edge_on_host([&](_util::edge_manager::edge_data<
                                           edge_capacity, data_capacity> &e) {
    using PointPartialDerivativesType =
        specfem::point::partial_derivatives<specfem::dimension::type::dim2,
                                            true, false>;
    e.ngll = qp5.NGLL;
    int ix, iz;
    int ispec = e.parent.id;
    for (int i = 0; i < qp5.NGLL; i++) {
      point_from_id(ix, iz, e.parent.bdry, i);
      e.x[i] = assembly->mesh.points.coord(0, ispec, iz, ix);
      e.z[i] = assembly->mesh.points.coord(1, ispec, iz, ix);
      specfem::point::index<specfem::dimension::type::dim2> index(ispec, iz,
                                                                  ix);

      PointPartialDerivativesType ppd;
      specfem::compute::load_on_host(index, assembly->partial_derivatives, ppd);
      // type_real dvdxi = assembly->mesh.quadratures.h_hprime(ix,ix);
      // type_real dvdga = assembly->mesh.quadratures.h_hprime(iz,iz);
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
        type_real dvdxi = assembly->mesh.quadratures.gll.h_hprime(ix, ixshape) *
                          (iz == izshape);
        type_real dvdga = assembly->mesh.quadratures.gll.h_hprime(iz, izshape) *
                          (ix == ixshape);
        type_real dvdx = dvdxi * ppd.xix + dvdga * ppd.gammax;
        type_real dvdz = dvdxi * ppd.xiz + dvdga * ppd.gammaz;
        e.data[EDGEIND_SHAPENDERIV + ishape][i] = (dvdx * nx + dvdz * nz) * det;
      }
      e.data[EDGEIND_BDRY_TYPE][i] =
          (type_real)assembly->boundaries.boundary_tags(ispec);
      specfem::point::boundary<specfem::element::boundary_tag::none,
                               specfem::dimension::type::dim2, false>
          bdnone;
      specfem::compute::load_on_host(index, assembly->boundaries, bdnone);
      e.data[EDGEIND_BDRY_TYPE + 1][i] =
          (type_real)(bdnone.tag == specfem::element::boundary_tag::none);
      specfem::point::boundary<
          specfem::element::boundary_tag::acoustic_free_surface,
          specfem::dimension::type::dim2, false>
          bdafs;
      specfem::compute::load_on_host(index, assembly->boundaries, bdafs);
      e.data[EDGEIND_BDRY_TYPE + 2][i] =
          (type_real)(bdafs.tag ==
                      specfem::element::boundary_tag::acoustic_free_surface);
    }
  });
  auto spec_charlen2 = [&](int ispec) {
    type_real lx1 =
        assembly->mesh.points.coord(0, ispec, 0, 0) -
        assembly->mesh.points.coord(0, ispec, qp5.NGLL - 1, qp5.NGLL - 1);
    type_real lz1 =
        assembly->mesh.points.coord(1, ispec, 0, 0) -
        assembly->mesh.points.coord(1, ispec, qp5.NGLL - 1, qp5.NGLL - 1);
    type_real lx2 = assembly->mesh.points.coord(0, ispec, qp5.NGLL - 1, 0) -
                    assembly->mesh.points.coord(0, ispec, 0, qp5.NGLL - 1);
    type_real lz2 = assembly->mesh.points.coord(1, ispec, qp5.NGLL - 1, 0) -
                    assembly->mesh.points.coord(1, ispec, 0, qp5.NGLL - 1);
    return std::max(lx1 * lx1 + lz1 * lz1, lx2 * lx2 + lz2 * lz2);
  };
  dg_edge_storage.foreach_intersection_on_host(
      [&](_util::edge_manager::edge_intersection<edge_capacity> &intersect,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
          _util::edge_manager::edge_data<edge_capacity, data_capacity> &b) {
        using AcousticProperties = specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::acoustic,
            specfem::element::property_tag::isotropic, false>;
        using ElasticProperties = specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic,
            specfem::element::property_tag::isotropic, false>;
        int a_ispec = a.parent.id;
        int b_ispec = b.parent.id;
        specfem::element::medium_tag a_medium =
            assembly->properties.h_element_types(a_ispec);
        specfem::element::medium_tag b_medium =
            assembly->properties.h_element_types(b_ispec);
        if (a_medium == specfem::element::medium_tag::acoustic &&
            b_medium == specfem::element::medium_tag::acoustic) {
          type_real rho_inv_max = 0;
          AcousticProperties props;
          for (int iz = 0; iz < qp5.NGLL; iz++) {
            for (int ix = 0; ix < qp5.NGLL; ix++) {
              specfem::point::index<specfem::dimension::type::dim2> index(
                  a_ispec, iz, ix);
              specfem::compute::load_on_host(index, assembly->properties,
                                             props);
              rho_inv_max = std::max(rho_inv_max, props.rho_inverse);
              index.ispec = b_ispec;
              specfem::compute::load_on_host(index, assembly->properties,
                                             props);
              rho_inv_max = std::max(rho_inv_max, props.rho_inverse);
            }
          }
          intersect.relax_param =
              _RELAX_PARAM_COEF_ACOUSTIC_ * rho_inv_max /
              sqrt(std::max(spec_charlen2(a_ispec), spec_charlen2(b_ispec)));

        } else if (a_medium == specfem::element::medium_tag::elastic &&
                   b_medium == specfem::element::medium_tag::acoustic) {
          // swap a and b
          int tmpi;
          type_real tmpr;
#define swpi(a, b)                                                             \
  {                                                                            \
    tmpi = a;                                                                  \
    a = b;                                                                     \
    b = tmpi;                                                                  \
  }
#define swpr(a, b)                                                             \
  {                                                                            \
    tmpr = a;                                                                  \
    a = b;                                                                     \
    b = tmpr;                                                                  \
  }
          swpi(intersect.a_ref_ind, intersect.b_ref_ind);
          swpi(intersect.a_ngll, intersect.b_ngll);
          swpr(intersect.a_param_start, intersect.b_param_start);
          swpr(intersect.a_param_end, intersect.b_param_end);
          for (int igll1 = 0; igll1 < edge_capacity; igll1++) {
            for (int igll2 = 0; igll2 < edge_capacity; igll2++) {
              swpr(intersect.a_mortar_trans[igll1][igll2],
                   intersect.b_mortar_trans[igll1][igll2])
            }
          }

#undef swpi
#undef swpr
        } else if (a_medium == specfem::element::medium_tag::acoustic &&
                   b_medium == specfem::element::medium_tag::elastic) {
          // this elif here just to branch away from the fail case
        } else if (a_medium == specfem::element::medium_tag::elastic &&
                   b_medium == specfem::element::medium_tag::elastic) {
          ElasticProperties props;

          type_real speed_param_max = 0;
          for (int iz = 0; iz < qp5.NGLL; iz++) {
            for (int ix = 0; ix < qp5.NGLL; ix++) {
              specfem::point::index<specfem::dimension::type::dim2> index(
                  a_ispec, iz, ix);
              specfem::compute::load_on_host(index, assembly->properties,
                                             props);
              speed_param_max = std::max(speed_param_max, props.lambdaplus2mu);
              index.ispec = b_ispec;
              specfem::compute::load_on_host(index, assembly->properties,
                                             props);
              speed_param_max = std::max(speed_param_max, props.lambdaplus2mu);
            }
          }
          intersect.relax_param =
              _RELAX_PARAM_COEF_ELASTIC_ * speed_param_max /
              sqrt(std::max(spec_charlen2(a_ispec), spec_charlen2(b_ispec)));
        } else {
          throw std::runtime_error(
              "Mortar Flux: medium combination not supported.");
        }
      });

  dg_edge_storage.initialize_intersection_data(intersect_data_capacity);

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

          using AcousticProperties = specfem::point::properties<
              specfem::dimension::type::dim2,
              specfem::element::medium_tag::acoustic,
              specfem::element::property_tag::isotropic, false>;
          using ElasticProperties = specfem::point::properties<
              specfem::dimension::type::dim2,
              specfem::element::medium_tag::elastic,
              specfem::element::property_tag::isotropic, false>;
          int ix, iz;
          int ispec = e.parent.id;
          for (int i = 0; i < qp5.NGLL; i++) {
            point_from_id(ix, iz, e.parent.bdry, i);
            specfem::point::index<specfem::dimension::type::dim2> index(ispec,
                                                                        iz, ix);

            PointPartialDerivativesType ppd;
            specfem::compute::load_on_host(index, assembly->partial_derivatives,
                                           ppd);
            int ncomp;
            std::vector<type_real> dfdxi;
            std::vector<type_real> dfdga;
            specfem::element::medium_tag medium =
                assembly->properties.h_element_types(ispec);
            if (medium == specfem::element::medium_tag::acoustic) {
              PointAcoustic disp;
              AcousticProperties props;
              constexpr int medium_ID =
                  static_cast<int>(specfem::element::medium_tag::acoustic);
              ncomp = 1;
              dfdxi = std::vector<type_real>(ncomp, 0);
              dfdga = std::vector<type_real>(ncomp, 0);
              specfem::compute::load_on_host(index, assembly->fields.forward,
                                             disp);
              specfem::compute::load_on_host(index, assembly->properties,
                                             props);
              e.data[EDGEIND_FIELD][i] = disp.displacement(0);
              e.data[EDGEIND_SPEEDPARAM][i] = props.rho_inverse;
              for (int k = 0; k < qp5.NGLL; k++) {
                specfem::point::index<specfem::dimension::type::dim2> index_(
                    ispec, iz, k);
                specfem::compute::load_on_host(index_, assembly->fields.forward,
                                               disp);
                dfdxi[0] += assembly->mesh.quadratures.gll.h_hprime(ix, k) *
                            disp.displacement(0);
                index_ = specfem::point::index<specfem::dimension::type::dim2>(
                    ispec, k, ix);
                specfem::compute::load_on_host(index_, assembly->fields.forward,
                                               disp);
                dfdga[0] += assembly->mesh.quadratures.gll.h_hprime(iz, k) *
                            disp.displacement(0);
              }
            } else if (medium == specfem::element::medium_tag::elastic) {
              PointElastic disp;
              ncomp = 2;
              constexpr int medium_ID =
                  static_cast<int>(specfem::element::medium_tag::elastic);
              dfdxi = std::vector<type_real>(ncomp, 0);
              dfdga = std::vector<type_real>(ncomp, 0);
              specfem::compute::load_on_host(index, assembly->fields.forward,
                                             disp);

              e.data[EDGEIND_FIELD][i] = disp.displacement(0);
              e.data[EDGEIND_FIELD + 1][i] = disp.displacement(1);
              for (int k = 0; k < qp5.NGLL; k++) {
                specfem::point::index<specfem::dimension::type::dim2> index_(
                    ispec, iz, k);
                specfem::compute::load_on_host(index_, assembly->fields.forward,
                                               disp);
                dfdxi[0] += assembly->mesh.quadratures.gll.h_hprime(ix, k) *
                            disp.displacement(0);
                dfdxi[1] += assembly->mesh.quadratures.gll.h_hprime(ix, k) *
                            disp.displacement(1);
                index_ = specfem::point::index<specfem::dimension::type::dim2>(
                    ispec, k, ix);
                specfem::compute::load_on_host(index_, assembly->fields.forward,
                                               disp);
                dfdga[0] += assembly->mesh.quadratures.gll.h_hprime(iz, k) *
                            disp.displacement(0);
                dfdga[1] += assembly->mesh.quadratures.gll.h_hprime(iz, k) *
                            disp.displacement(1);
              }
              e.data[EDGEIND_SPEEDPARAM][i] =
                  assembly->properties.acoustic_isotropic.h_rho_inverse(ispec,
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

  const auto h_is_bdry_at_pt =
      [&](const specfem::point::index<specfem::dimension::type::dim2> index,
          specfem::element::boundary_tag tag) -> bool {
    specfem::point::boundary<
        specfem::element::boundary_tag::acoustic_free_surface,
        specfem::dimension::type::dim2, false>
        point_boundary_afs;
    specfem::point::boundary<specfem::element::boundary_tag::none,
                             specfem::dimension::type::dim2, false>
        point_boundary_none;
    specfem::point::boundary<specfem::element::boundary_tag::stacey,
                             specfem::dimension::type::dim2, false>
        point_boundary_stacey;
    specfem::point::boundary<
        specfem::element::boundary_tag::composite_stacey_dirichlet,
        specfem::dimension::type::dim2, false>
        point_boundary_composite;
    switch (assembly->boundaries.boundary_tags(index.ispec)) {
    case specfem::element::boundary_tag::acoustic_free_surface:
      specfem::compute::load_on_device(index, assembly->boundaries,
                                       point_boundary_afs);
      return point_boundary_afs.tag == tag;
    case specfem::element::boundary_tag::none:
      specfem::compute::load_on_device(index, assembly->boundaries,
                                       point_boundary_none);
      return point_boundary_none.tag == tag;
    case specfem::element::boundary_tag::stacey:
      specfem::compute::load_on_device(index, assembly->boundaries,
                                       point_boundary_stacey);
      return point_boundary_stacey.tag == tag;
    case specfem::element::boundary_tag::composite_stacey_dirichlet:
      specfem::compute::load_on_device(index, assembly->boundaries,
                                       point_boundary_composite);
      return point_boundary_composite.tag == tag;
    default:
      throw std::runtime_error(
          "h_is_bdry_at_pt: unknown assembly->boundaries.boundary_tags(ispec) "
          "value!");
      return false;
    }
  };

  specfem::event_marching::arbitrary_call_event mortar_flux_acoustic(
      [&]() {
        dg_edge_storage.foreach_intersection_on_host(
            [&](_util::edge_manager::edge_intersection<edge_capacity>
                    &intersect,
                _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
                _util::edge_manager::edge_data<edge_capacity, data_capacity> &b,
                decltype(Kokkos::subview(
                    std::declval<specfem::kokkos::HostView2d<type_real> >(), 1u,
                    Kokkos::ALL)) data_view) {
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
                  assembly->properties.h_element_types(a_ispec);
              specfem::element::medium_tag b_medium =
                  assembly->properties.h_element_types(b_ispec);
              type_real a_scale_dS =
                  0.5 * (intersect.a_param_end - intersect.a_param_start);
              type_real b_scale_dS =
                  0.5 * (intersect.b_param_end - intersect.b_param_start);
              // currently, edge_manager assumes any acoustic - x boundaries
              // have acoustic on the a-side.
              if (a_medium == specfem::element::medium_tag::acoustic) {
                if (b_medium == specfem::element::medium_tag::acoustic) {
                  for (int a_ishape = 0; a_ishape < a.ngll; a_ishape++) {
                    // flux += (
                    //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                    //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                    //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                    // )
                    type_real flux = 0;
                    type_real f1 = 0;
                    type_real f2 = 0;
                    type_real f3 = 0;
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
                                         iquad, b.data[EDGEIND_FIELDNDERIV]) *
                                     (b_scale_dS / a_scale_dS));
                      data_view[INTERIND_UJMP + iquad] = u_jmp;
                      data_view[INTERIND_DU_AVG + iquad] = cdu_avg;

                      f1 += a_scale_dS * gll.w[iquad] * 0.5 *
                            intersect.a_to_mortar(
                                iquad, a.data[EDGEIND_SHAPENDERIV + a_ishape]) *
                            c * u_jmp;
                      f2 += a_scale_dS * gll.w[iquad] *
                            intersect.a_mortar_trans[iquad][a_ishape] *
                            (cdu_avg);
                      f3 += a_scale_dS * gll.w[iquad] *
                            intersect.a_to_mortar(iquad, a.data[EDGEIND_DS]) *
                            intersect.a_mortar_trans[iquad][a_ishape] *
                            (-intersect.relax_param * u_jmp);
                      flux +=
                          a_scale_dS * gll.w[iquad] *
                          (0.5 *
                               intersect.a_to_mortar(
                                   iquad,
                                   a.data[EDGEIND_SHAPENDERIV + a_ishape]) *
                               c * u_jmp +
                           intersect.a_mortar_trans[iquad][a_ishape] *
                               (cdu_avg - intersect.a_to_mortar(
                                              iquad, a.data[EDGEIND_DS]) *
                                              intersect.relax_param * u_jmp));
                    }
                    data_view[INTERIND_FLUXTOTAL_A + a_ishape] = flux;
                    data_view[INTERIND_FLUX1_A + a_ishape] = f1;
                    data_view[INTERIND_FLUX2_A + a_ishape] = f2;
                    data_view[INTERIND_FLUX3_A + a_ishape] = f3;
                    PointAcousticAccel accel;
                    specfem::point::index<specfem::dimension::type::dim2> index(
                        a_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, a.parent.bdry, a_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration = flux;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 1;
                    }
                  }
                  for (int b_ishape = 0; b_ishape < b.ngll; b_ishape++) {
                    // flux += (
                    //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                    //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                    //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                    // )
                    type_real flux = 0;
                    type_real f1 = 0;
                    type_real f2 = 0;
                    type_real f3 = 0;
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
                                         iquad, a.data[EDGEIND_FIELDNDERIV]) *
                                     (a_scale_dS / b_scale_dS));
                      f1 += b_scale_dS * gll.w[iquad] * 0.5 *
                            intersect.b_to_mortar(
                                iquad, b.data[EDGEIND_SHAPENDERIV + b_ishape]) *
                            c * u_jmp;
                      f2 += b_scale_dS * gll.w[iquad] *
                            intersect.b_mortar_trans[iquad][b_ishape] *
                            (cdu_avg);
                      f3 += b_scale_dS * gll.w[iquad] *
                            intersect.b_to_mortar(iquad, b.data[EDGEIND_DS]) *
                            intersect.b_mortar_trans[iquad][b_ishape] *
                            (-intersect.relax_param * u_jmp);
                      flux +=
                          b_scale_dS * gll.w[iquad] *
                          (0.5 *
                               intersect.b_to_mortar(
                                   iquad,
                                   b.data[EDGEIND_SHAPENDERIV + b_ishape]) *
                               c * u_jmp +
                           intersect.b_mortar_trans[iquad][b_ishape] *
                               (cdu_avg - intersect.b_to_mortar(
                                              iquad, b.data[EDGEIND_DS]) *
                                              intersect.relax_param * u_jmp));
                    }
                    data_view[INTERIND_FLUXTOTAL_B + b_ishape] = flux;
                    data_view[INTERIND_FLUX1_B + b_ishape] = f1;
                    data_view[INTERIND_FLUX2_B + b_ishape] = f2;
                    data_view[INTERIND_FLUX3_B + b_ishape] = f3;
                    PointAcousticAccel accel;
                    specfem::point::index<specfem::dimension::type::dim2> index(
                        b_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, b.parent.bdry, b_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration = flux;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 1;
                    }
                  }

                } else if (b_medium == specfem::element::medium_tag::elastic) {
                  // if(a_ispec == 82 && (a.data[EDGEIND_FIELD][0] > 0.001 ||
                  // a.data[EDGEIND_FIELD][4] > 0.001)){
                  if (a_ispec == 53 && timescheme_wrapper.get_istep() == 6) {
                    intersect.b_to_mortar(
                        0, b.data[EDGEIND_FIELD]); // place breakpoint here
                  }
                  for (int a_ishape = 0; a_ishape < a.ngll; a_ishape++) {
                    // flux = JW * v * s.n
                    type_real flux = 0;
                    for (int iquad = 0; iquad < gll.nquad; iquad++) {
                      type_real sn_dSb =
                          -( // yes this is not optimized.
                              intersect.b_to_mortar(iquad,
                                                    b.data[EDGEIND_FIELD]) *
                                  intersect.b_to_mortar(iquad,
                                                        b.data[EDGEIND_NX]) +
                              intersect.b_to_mortar(iquad,
                                                    b.data[EDGEIND_FIELD + 1]) *
                                  intersect.b_to_mortar(iquad,
                                                        b.data[EDGEIND_NZ])) *
                          intersect.b_to_mortar(iquad, b.data[EDGEIND_DET]);
                      flux += b_scale_dS * gll.w[iquad] * sn_dSb *
                              intersect.a_mortar_trans[iquad][a_ishape];
                    }
                    data_view[INTERIND_FLUXTOTAL_A + a_ishape] = flux;
                    PointAcousticAccel accel;
                    specfem::point::index<specfem::dimension::type::dim2> index(
                        a_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, a.parent.bdry, a_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration = flux;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 1;
                    }
                  }
                } else {
                  throw std::runtime_error(
                      "Mortar Flux: medium combination not supported.");
                }
              }
            });
        assembly->fields.forward.copy_to_device();
        return 0;
      },
      0.91);
  timescheme_wrapper.time_stepper.register_event(&mortar_flux_acoustic);

  specfem::event_marching::arbitrary_call_event mortar_flux_elastic(
      [&]() {
        dg_edge_storage.foreach_intersection_on_host(
            [&](_util::edge_manager::edge_intersection<edge_capacity>
                    &intersect,
                _util::edge_manager::edge_data<edge_capacity, data_capacity> &a,
                _util::edge_manager::edge_data<edge_capacity, data_capacity> &b,
                decltype(Kokkos::subview(
                    std::declval<specfem::kokkos::HostView2d<type_real> >(), 1u,
                    Kokkos::ALL)) data_view) {
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
                  assembly->properties.h_element_types(a_ispec);
              specfem::element::medium_tag b_medium =
                  assembly->properties.h_element_types(b_ispec);
              if ((a_ispec == 88 || b_ispec == 88) &&
                  timescheme_wrapper.get_istep() == 1000) {
                intersect.a_to_mortar(
                    1, a.data[EDGEIND_FIELDNDERIV]); // place breakpoint here
              }
              type_real a_scale_dS =
                  0.5 * (intersect.a_param_end - intersect.a_param_start);
              type_real b_scale_dS =
                  0.5 * (intersect.b_param_end - intersect.b_param_start);
              if (b_medium == specfem::element::medium_tag::elastic) {
                if (a_medium == specfem::element::medium_tag::elastic) {
                  throw std::runtime_error(
                      "Mortar Flux: elastic-elastic not yet supported.");
                  for (int a_ishape = 0; a_ishape < a.ngll; a_ishape++) {
                    // flux += (
                    //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                    //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                    //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                    // )
                    type_real flux = 0;
                    type_real f1 = 0;
                    type_real f2 = 0;
                    type_real f3 = 0;
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
                                         iquad, b.data[EDGEIND_FIELDNDERIV]) *
                                     (b_scale_dS / a_scale_dS));
                      data_view[INTERIND_UJMP + iquad] = u_jmp;
                      data_view[INTERIND_DU_AVG + iquad] = cdu_avg;

                      f1 += a_scale_dS * gll.w[iquad] * 0.5 *
                            intersect.a_to_mortar(
                                iquad, a.data[EDGEIND_SHAPENDERIV + a_ishape]) *
                            c * u_jmp;
                      f2 += a_scale_dS * gll.w[iquad] *
                            intersect.a_mortar_trans[iquad][a_ishape] *
                            (cdu_avg);
                      f3 += a_scale_dS * gll.w[iquad] *
                            intersect.a_to_mortar(iquad, a.data[EDGEIND_DS]) *
                            intersect.a_mortar_trans[iquad][a_ishape] *
                            (-intersect.relax_param * u_jmp);
                      flux +=
                          a_scale_dS * gll.w[iquad] *
                          (0.5 *
                               intersect.a_to_mortar(
                                   iquad,
                                   a.data[EDGEIND_SHAPENDERIV + a_ishape]) *
                               c * u_jmp +
                           intersect.a_mortar_trans[iquad][a_ishape] *
                               (cdu_avg - intersect.a_to_mortar(
                                              iquad, a.data[EDGEIND_DS]) *
                                              intersect.relax_param * u_jmp));
                    }
                    data_view[INTERIND_FLUXTOTAL_A + a_ishape] = flux;
                    data_view[INTERIND_FLUX1_A + a_ishape] = f1;
                    data_view[INTERIND_FLUX2_A + a_ishape] = f2;
                    data_view[INTERIND_FLUX3_A + a_ishape] = f3;
                    PointAcousticAccel accel;
                    specfem::point::index<specfem::dimension::type::dim2> index(
                        a_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, a.parent.bdry, a_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration = flux;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_A + a_ishape] = 1;
                    }
                  }
                  for (int b_ishape = 0; b_ishape < b.ngll; b_ishape++) {
                    // flux += (
                    //       np.einsum("j,j,j,ji->i",JW,u-u_,c/2,dv)
                    //     + np.einsum("j,ji,j->i",JW,v,0.5*(c*du+c_*du_))
                    //     - a*np.einsum("j,j,ji->i",JW,u-u_,v)
                    // )
                    type_real flux = 0;
                    type_real f1 = 0;
                    type_real f2 = 0;
                    type_real f3 = 0;
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
                                         iquad, a.data[EDGEIND_FIELDNDERIV]) *
                                     (a_scale_dS / b_scale_dS));
                      f1 += b_scale_dS * gll.w[iquad] * 0.5 *
                            intersect.b_to_mortar(
                                iquad, b.data[EDGEIND_SHAPENDERIV + b_ishape]) *
                            c * u_jmp;
                      f2 += b_scale_dS * gll.w[iquad] *
                            intersect.b_mortar_trans[iquad][b_ishape] *
                            (cdu_avg);
                      f3 += b_scale_dS * gll.w[iquad] *
                            intersect.b_to_mortar(iquad, b.data[EDGEIND_DS]) *
                            intersect.b_mortar_trans[iquad][b_ishape] *
                            (-intersect.relax_param * u_jmp);
                      flux +=
                          b_scale_dS * gll.w[iquad] *
                          (0.5 *
                               intersect.b_to_mortar(
                                   iquad,
                                   b.data[EDGEIND_SHAPENDERIV + b_ishape]) *
                               c * u_jmp +
                           intersect.b_mortar_trans[iquad][b_ishape] *
                               (cdu_avg - intersect.b_to_mortar(
                                              iquad, b.data[EDGEIND_DS]) *
                                              intersect.relax_param * u_jmp));
                    }
                    data_view[INTERIND_FLUXTOTAL_B + b_ishape] = flux;
                    data_view[INTERIND_FLUX1_B + b_ishape] = f1;
                    data_view[INTERIND_FLUX2_B + b_ishape] = f2;
                    data_view[INTERIND_FLUX3_B + b_ishape] = f3;
                    PointAcousticAccel accel;
                    specfem::point::index<specfem::dimension::type::dim2> index(
                        b_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, b.parent.bdry, b_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration = flux;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 1;
                    }
                  }

                } else if (a_medium == specfem::element::medium_tag::acoustic) {
                  if (a_ispec == 82 && (a.data[EDGEIND_FIELD][0] > 0.001 ||
                                        a.data[EDGEIND_FIELD][4] > 0.001)) {
                    intersect.b_to_mortar(
                        0, b.data[EDGEIND_FIELD]); // place breakpoint here
                  }
                  type_real a_accel[edge_capacity];
                  PointAcousticAccel a_accel_f;
                  specfem::point::index<specfem::dimension::type::dim2> index(
                      a_ispec, 0, 0);
                  for (int a_ishape = 0; a_ishape < a.ngll; a_ishape++) {
                    point_from_id(index.ix, index.iz, a.parent.bdry, a_ishape);
                    specfem::compute::load_on_device(
                        index, assembly->fields.forward, a_accel_f);
                    a_accel[a_ishape] = a_accel_f.acceleration(0);
                  }
                  for (int b_ishape = 0; b_ishape < b.ngll; b_ishape++) {
                    // flux = JW * v * chi_tt
                    type_real flux_x = 0;
                    type_real flux_z = 0;
                    for (int iquad = 0; iquad < gll.nquad; iquad++) {
                      type_real chitt_v_Ja =
                          -intersect.a_to_mortar(iquad, a.data[EDGEIND_DET]) *
                          intersect.a_to_mortar(iquad, a_accel) *
                          intersect.b_mortar_trans[iquad][b_ishape];
                      flux_x +=
                          a_scale_dS * gll.w[iquad] *
                          intersect.a_to_mortar(iquad, a.data[EDGEIND_NX]) *
                          chitt_v_Ja;
                      flux_z +=
                          a_scale_dS * gll.w[iquad] *
                          intersect.a_to_mortar(iquad, a.data[EDGEIND_NZ]) *
                          chitt_v_Ja;
                    }
                    data_view[INTERIND_FLUXTOTAL_B + b_ishape] = flux_z;
                    PointElasticAccel accel;
                    index =
                        specfem::point::index<specfem::dimension::type::dim2>(
                            b_ispec, 0, 0);
                    point_from_id(index.ix, index.iz, b.parent.bdry, b_ishape);
                    if (h_is_bdry_at_pt(index,
                                        specfem::element::boundary_tag::none)) {
                      accel.acceleration(0) = flux_x;
                      accel.acceleration(1) = flux_z;
                      specfem::compute::atomic_add_on_device(
                          index, accel, assembly->fields.forward);
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 0;
                    } else {
                      data_view[INTERIND_ACCEL_INCLUDE_B + b_ishape] = 1;
                    }
                  }
                } else {
                  throw std::runtime_error(
                      "Mortar Flux: medium combination not supported.");
                }
              }
            });
        assembly->fields.forward.copy_to_device();
        return 0;
      },
      1.1);
  timescheme_wrapper.time_stepper.register_event(&mortar_flux_elastic);

  specfem::event_marching::arbitrary_call_event write_outputs_at_end(
      [&]() {
        params.set_writers_from_runtime_configuration();
        for (auto writer : params.get_writers()) {
          if (writer) {
            mpi->cout("Writing Output:");
            mpi->cout("-------------------------------");

            writer->write();
          }
        }
        return 0;
      },
      2);
  event_system.register_event(&write_outputs_at_end);

  // init kernels needs to happen as of time_marcher
  kernels.initialize(timescheme.get_timestep());

  // ================================RUNTIME
  // BEGIN=================================
  event_system.run();
#undef edge_capacity
#undef data_capacity
}

// includes for load_parameters from specfem2d.cpp
#include "compute/interface.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "IO/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "receiver/interface.hpp"
#include "solver/solver.hpp"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
// Specfem2d driver

_util::demo_assembly::simulation_params
load_parameters(const std::string &parameter_file, specfem::MPI::MPI *mpi) {
  // --------------------------------------------------------------
  //                    Read parameter file
  // --------------------------------------------------------------
  auto start_time = std::chrono::system_clock::now();
  std::shared_ptr<specfem::runtime_configuration::setup> setup_ptr =
      std::make_shared<specfem::runtime_configuration::setup>(parameter_file,
                                                              __default_file__);
#define setup (*setup_ptr)
  const auto [database_filename, source_filename] = setup.get_databases();
  mpi->cout(setup.print_header(start_time));

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read mesh and materials
  // --------------------------------------------------------------
  const auto quadrature = setup.instantiate_quadrature();
  const auto mesh = specfem::IO::read_mesh(database_filename, mpi);
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read Sources and Receivers
  // --------------------------------------------------------------
  const int nsteps = setup.get_nsteps();
  const specfem::simulation::type simulation_type = setup.get_simulation_type();
  auto [sources, t0] = specfem::IO::read_sources(
      source_filename, nsteps, setup.get_t0(), setup.get_dt(), simulation_type);
  setup.update_t0(t0); // Update t0 in case it was changed

  const auto stations_filename = setup.get_stations_file();
  const auto angle = setup.get_receiver_angle();
  auto receivers = specfem::IO::read_receivers(stations_filename, angle);

  mpi->cout("Source Information:");
  mpi->cout("-------------------------------");
  if (mpi->main_proc()) {
    std::cout << "Number of sources : " << sources.size() << "\n" << std::endl;
  }

  for (auto &source : sources) {
    mpi->cout(source->print());
  }

  mpi->cout("Receiver Information:");
  mpi->cout("-------------------------------");

  if (mpi->main_proc()) {
    std::cout << "Number of receivers : " << receivers.size() << "\n"
              << std::endl;
  }

  for (auto &receiver : receivers) {
    mpi->cout(receiver->print());
  }
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Instantiate Timescheme
  // --------------------------------------------------------------
  const auto time_scheme = setup.instantiate_timescheme();
  if (mpi->main_proc())
    std::cout << *time_scheme << std::endl;

  const int max_seismogram_time_step = time_scheme->get_max_seismogram_step();
  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Generate Assembly
  // --------------------------------------------------------------
  mpi->cout("Generating assembly:");
  mpi->cout("-------------------------------");
  const type_real dt = setup.get_dt();
  specfem::compute::assembly assembly(
      mesh, quadrature, sources, receivers, setup.get_seismogram_types(),
      setup.get_t0(), dt, nsteps, max_seismogram_time_step,
      setup.get_simulation_type());
  time_scheme->link_assembly(assembly);

  // --------------------------------------------------------------

  // --------------------------------------------------------------
  //                   Read wavefields
  // --------------------------------------------------------------

  const auto wavefield_reader = setup.instantiate_wavefield_reader(assembly);
  if (wavefield_reader) {
    mpi->cout("Reading wavefield files:");
    mpi->cout("-------------------------------");

    wavefield_reader->read();
    // Transfer the buffer field to device
    assembly.fields.buffer.copy_to_device();
  }

  _util::demo_assembly::simulation_params params;

  params.dt(dt)
      .nsteps(nsteps)
      .t0(setup.get_t0())
      .simulation_type(setup.get_simulation_type())
      .mesh(mesh)
      .quadrature(quadrature)
      .sources(sources)
      .receivers(receivers)
      .seismogram_types(setup.get_seismogram_types())
      .nseismogram_steps(max_seismogram_time_step)
      .assembly(assembly)
      .runtime_configuration(setup_ptr);
#undef setup
  return params;
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
