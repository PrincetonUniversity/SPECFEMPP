#include "SPECFEM_Environment.hpp"

#include <ios>
#include <stdexcept>

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.tpp"
#include "specfem_setup.hpp"
#include <gtest/gtest.h>

TEST(impl__compute_intersection, TransferFunctionCorrectness) {
  const int ngnod = 4;
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg1("coorg1", ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg2("coorg2", ngnod);
  // element 1 lies on [0,1] x [0,1]
  coorg1(0) = { 0, 0 };
  coorg1(1) = { 1, 0 };
  coorg1(2) = { 1, 1 };
  coorg1(3) = { 0, 1 };

  // match element 1 (right) to element 2 (left)
  coorg2(0).x = 1;
  coorg2(1).x = 2;
  coorg2(2).x = 2;
  coorg2(3).x = 1;

  // use some weird quadrature rule on mortar -- we want a higher degree so
  // collocation is exact
  const int nquad_mortar = 5;
  const Kokkos::View<type_real *, Kokkos::HostSpace> mortar_quad("mortar_quad",
                                                                 nquad_mortar);
  mortar_quad(0) = -1;
  mortar_quad(1) = -0.7;
  mortar_quad(2) = 0;
  mortar_quad(3) = 0.7;
  mortar_quad(4) = 1;

  // some other rule on edge -- should be lower (or equal) degree than
  // mortar_quad
  const int nquad_edge = 4;
  const Kokkos::View<type_real *, Kokkos::HostSpace> edge_quad("edge_quad",
                                                               nquad_edge);
  edge_quad(0) = -1;
  edge_quad(1) = 0;
  edge_quad(2) = 0.33;
  edge_quad(3) = 1;

  // singlecall- set_transfer_functions without deriv
  // bothcall- set_transfer_functions with deriv
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans1_singlecall(
      "mortar_trans1_singlecall", nquad_mortar, nquad_edge);
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans1_bothcall(
      "mortar_trans1_bothcall", nquad_mortar, nquad_edge);
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans1p(
      "mortar_trans1p", nquad_mortar, nquad_edge);
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans2_singlecall(
      "mortar_trans2_singlecall", nquad_mortar, nquad_edge);
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans2_bothcall(
      "mortar_trans2_bothcall", nquad_mortar, nquad_edge);
  Kokkos::View<type_real **, Kokkos::HostSpace> mortar_trans2p(
      "mortar_trans2p", nquad_mortar, nquad_edge);

  // different vertical offsets for element 2
  for (const auto [coord_lo, coord_hi] :
       std::vector<std::pair<type_real, type_real> >{
           { 0, 1 }, { -1, 0.5 }, { 0.5, 1 }, { 1.5, 2.5 }, { -100, -0.1 } }) {
    coorg2(0).z = coord_lo;
    coorg2(1).z = coord_lo;
    coorg2(2).z = coord_hi;
    coorg2(3).z = coord_hi;

    if (coord_lo > 1 || coord_hi < 0) {
      EXPECT_THROW(specfem::assembly::nonconforming_interfaces_impl::
                       set_transfer_functions(
                           coorg1, coorg2,
                           specfem::mesh_entity::dim2::type::right,
                           specfem::mesh_entity::dim2::type::left, mortar_quad,
                           edge_quad, mortar_trans1_singlecall,
                           mortar_trans2_singlecall),
                   std::runtime_error)
          << "Global coordinate intervals:\n"
          << "   side 1: [0, 1]\n"
          << "   side 2: [" << coord_lo << ", " << coord_hi << "]\n"
          << "There should be no intersection, causing `compute_intersection` "
             "to throw an error, but none was thrown.";
      EXPECT_THROW(specfem::assembly::nonconforming_interfaces_impl::
                       set_transfer_functions(
                           coorg1, coorg2,
                           specfem::mesh_entity::dim2::type::right,
                           specfem::mesh_entity::dim2::type::left, mortar_quad,
                           edge_quad, mortar_trans1_bothcall, mortar_trans1p,
                           mortar_trans2_bothcall, mortar_trans2p),
                   std::runtime_error)
          << "Global coordinate intervals:\n"
          << "   side 1: [0, 1]\n"
          << "   side 2: [" << coord_lo << ", " << coord_hi << "]\n"
          << "There should be no intersection, causing `compute_intersection` "
             "to throw an error, but none was thrown.";
      continue;
    }

    const type_real eps = 1e-3;

    specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
        coorg1, coorg2, specfem::mesh_entity::dim2::type::right,
        specfem::mesh_entity::dim2::type::left, mortar_quad, edge_quad,
        mortar_trans1_singlecall, mortar_trans2_singlecall);
    specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
        coorg1, coorg2, specfem::mesh_entity::dim2::type::right,
        specfem::mesh_entity::dim2::type::left, mortar_quad, edge_quad,
        mortar_trans1_bothcall, mortar_trans1p, mortar_trans2_bothcall,
        mortar_trans2p);

    // edge shape function jquad evaluated at mortar knot imort:
    // mortar_trans[imort, jquad]
    // interpolate this function at knot<i>_mortar and see if we get
    // kronecker delta{i,j}
    {
      bool failed_test = false;
      std::ostringstream failmsg;

      const type_real global_lo = std::max(coord_lo, type_real(0));
      const type_real global_hi = std::min(coord_hi, type_real(1));

      // jacobian of edge->mortar reparameterization
      const type_real dmortar_dedge1 = (global_hi - global_lo) / (1 - 0);
      const type_real dmortar_dedge2 =
          (global_hi - global_lo) / (coord_hi - coord_lo);
      // mortar -> global mapping:
      //   -1 -> global_lo
      //    1 -> global_hi
      for (int iquad = 0; iquad < nquad_edge; iquad++) {
        const type_real knot1_global = (1 + edge_quad(iquad)) / 2;
        const type_real knot2_global =
            ((coord_lo + coord_hi) + edge_quad(iquad) * (coord_hi - coord_lo)) /
            2;
        // ??? -> knot<i>_global
        const type_real knot1_mortar =
            (2 * knot1_global - (global_hi + global_lo)) /
            (global_hi - global_lo);
        const type_real knot2_mortar =
            (2 * knot2_global - (global_hi + global_lo)) /
            (global_hi - global_lo);

        const auto [mortar_shape_at_edge1, dmortar_shape_at_edge1] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                knot1_mortar, nquad_mortar, mortar_quad);
        const auto [mortar_shape_at_edge2, dmortar_shape_at_edge2] =
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                knot2_mortar, nquad_mortar, mortar_quad);

        const auto dedge_shape = std::get<1>(
            specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
                edge_quad(iquad), nquad_edge, edge_quad));

        for (int jquad = 0; jquad < nquad_edge; jquad++) {

          for (int side = 1; side <= 2; side++) {
            const auto &mortar_shape_at_edge =
                (side == 1) ? mortar_shape_at_edge1 : mortar_shape_at_edge2;
            const auto &dmortar_shape_at_edge =
                (side == 1) ? dmortar_shape_at_edge1 : dmortar_shape_at_edge2;
            const auto &mortar_trans =
                (side == 1) ? mortar_trans1_bothcall : mortar_trans2_bothcall;
            const auto &mortar_transp =
                (side == 1) ? mortar_trans1p : mortar_trans2p;
            const auto &dmortar_dedge =
                (side == 1) ? dmortar_dedge1 : dmortar_dedge2;

            //  accumulate edge_shapefunction[jquad] evaluated at knot iquad:
            // (H[jquad]○to_edge○to_mortar)(knot[iquad]) ==
            // H[jquad](knot[iquad])
            //                     testing this equality ^^
            type_real accum = 0;

            // there are a few ways of getting H[jquad]':
            // interpolate H in mortar space, differentiating interpolants (in
            // mortar space) evaluated at edge knot (dmortar_shape_at_edge),
            // then chain-rule:
            type_real accum_dmortar = 0;
            // interpolate H' (mortar_trans) in mortar space:
            type_real accum_dedge = 0;

            // or just simply use H' (dedge_shape)
            for (int imort = 0; imort < nquad_mortar; imort++) {
              accum += mortar_shape_at_edge(imort) * mortar_trans(imort, jquad);
              accum_dmortar +=
                  dmortar_shape_at_edge(imort) * mortar_trans(imort, jquad);
              accum_dedge +=
                  mortar_shape_at_edge(imort) * mortar_transp(imort, jquad);
            }
            type_real expect = (iquad == jquad) ? 1 : 0;
            bool curfail = false;

            // shapefunc
            if (std::abs(accum - expect) > eps) {
              if (!curfail) {
                failmsg << "\n  Edge " << side << " -- H[" << jquad
                        << "] @ knot " << iquad << ":";
                curfail = true;
                failed_test = true;
              }
              failmsg << "\n    evaluates to " << accum << "(Should be "
                      << expect << ")";
            }
            // dshapefunc (test via chain rule -- this will not verify
            // mortar_transp)
            if (std::abs(accum_dmortar / dmortar_dedge - dedge_shape(jquad)) >
                eps) {
              if (!curfail) {
                failmsg << "\n  Edge " << side << " -- H[" << jquad
                        << "] @ knot " << iquad << ":";
                curfail = true;
                failed_test = true;
              }
              failmsg << "\n    derivative is " << accum_dmortar
                      << " w.r.t. mortar coordinate (Should be "
                      << dedge_shape(jquad) * dmortar_dedge
                      << "; mistake in mortar transfer, mortar-space "
                         "interpolant, or edge <-> mortar jacobian "
                      << dmortar_dedge << ")";
            }
            // dshapefunc (test via mortar_transp)
            if (std::abs(accum_dedge - dedge_shape(jquad)) > eps) {
              if (!curfail) {
                failmsg << "\n  Edge " << side << " -- H[" << jquad
                        << "] @ knot " << iquad << ":";
                curfail = true;
                failed_test = true;
              }
              failmsg << "\n    derivative is " << accum_dedge
                      << " w.r.t. mortar coordinate (Should be "
                      << dedge_shape(jquad) << ")";
            }
          }
        }
      }

      if (failed_test) {
        FAIL() << "Global coordinate intervals:\n"
               << "   side 1: [0, 1]\n"
               << "   side 2: [" << coord_lo << ", " << coord_hi << "]\n"
               << "   intersection: [" << global_lo << ", " << global_hi
               << "]\n"
               << "Failed edge->mortar->edge interpolation test:"
               << failmsg.str();
      }
    }

    // verify deriv inclusion / exclusion equivalent:
    {
      // TODO: later PR: nicer array check
      const type_real eps = 1e-7;
      bool still_fine = true;
      for (int iquad = 0; iquad < nquad_mortar && still_fine; iquad++) {
        for (int jquad = 0; jquad < nquad_edge && still_fine; jquad++) {

          if (std::abs(mortar_trans1_bothcall(iquad, jquad) -
                       mortar_trans1_singlecall(iquad, jquad)) > eps) {
            FAIL() << "set_transfer_functions() should be equivalent whether "
                      "or not a derivative is requested. On side 1, we found a "
                      "disagreement for shape function "
                   << jquad << " at mortar quadrature point " << iquad
                   << " (mortar coordinate " << mortar_quad(iquad) << ").\n"
                   << "  With derivative requested: "
                   << mortar_trans1_bothcall(iquad, jquad)
                   << "\n                    Without: "
                   << mortar_trans1_singlecall(iquad, jquad)
                   << "\n                      Error: " << std::scientific
                   << std::showpos
                   << mortar_trans1_bothcall(iquad, jquad) -
                          mortar_trans1_singlecall(iquad, jquad);

            still_fine = false;
          }
        }
      }
    }
  }
}
