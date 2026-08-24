
#include "acoustic_elastic.hpp"
#include "integrate_against_shape_on_face.hpp"
#include "specfem/compute/impl/compute_coupling.hpp"
#include "specfem/compute/impl/compute_coupling.tpp" // so that we don't need to load the entire solver
#include "specfem/element/attributes.hpp"
#include "specfem/element/dimension.hpp"
#include "specfem/element/tags.hpp"
#include "utilities/include/fieldmanip/fieldgetter.hpp"
#include "utilities/include/fieldmanip/fieldsetter.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>

// helper: hides away the specfem::compute::impl stuff.
template <int NGLL>
void acoustic_compute_update(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;

  specfem::compute::impl::compute_coupling<
      NGLL, specfem::tags::Tags<dimension_tag,
                                specfem::simulation::field_type::forward,
                                specfem::element::medium_tag::acoustic>>(
      assembly);
}

// helper: hides away the specfem::compute::impl stuff.
template <int NGLL>
void elastic_compute_update(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly) {
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;

  specfem::compute::impl::compute_coupling<
      NGLL, specfem::tags::Tags<dimension_tag,
                                specfem::simulation::field_type::forward,
                                specfem::element::medium_tag::elastic>>(
      assembly);
}

/**
 * @brief Sets the acoustic acceleration field to power function.
 *
 * when passed to specfem::test_fieldmanip::set_field_values(), sets
 * acceleration to (x/xscale)^{xpow} (y/yscale)^{ypow}. Zeroes out displacement
 * and velocity.
 */
template <int xpow, int ypow>
struct acoustic_field_initializer_pow
    : public specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::acoustic> {
  type_real inv_xscale;
  type_real inv_yscale;
  KOKKOS_INLINE_FUNCTION PointAccelerationType
  acceleration(const PointData &data) const {
    PointAccelerationType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = std::pow(data.coords.x * inv_xscale, xpow) *
                   std::pow(data.coords.y * inv_yscale, ypow);
    }
    return val;
  }
  acoustic_field_initializer_pow(const type_real &xscale,
                                 const type_real &yscale)
      : PointSetter(true, true, true), inv_xscale(1 / xscale),
        inv_yscale(1 / yscale) {};
};

/**
 * @brief Sets the elastic displacement field to power.
 *
 * when passed to specfem::test_fieldmanip::set_field_values(), sets
 * displacement to dir * (x/xscale)^{xpow} (y/yscale)^{ypow}, where dir is a
 * constant vector. Zeroes out displacement and velocity.
 */
template <int xpow, int ypow>
struct elastic_field_initializer_pow
    : public specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3,
          specfem::element::medium_tag::elastic> {
  type_real setdir[ndim];
  type_real inv_xscale;
  type_real inv_yscale;

  KOKKOS_INLINE_FUNCTION PointDisplacementType
  displacement(const PointData &data) const {
    PointDisplacementType val;
    constexpr int ncomp =
        specfem::element::attributes<dimension_tag, medium_tag>::components;
    type_real powpos = std::pow(data.coords.x * inv_xscale, xpow) *
                       std::pow(data.coords.y * inv_yscale, ypow);

    for (int icomp = 0; icomp < ncomp; icomp++) {
      val(icomp) = powpos * setdir[icomp];
    }
    return val;
  }
  elastic_field_initializer_pow(const type_real &xscale,
                                const type_real &yscale)
      : PointSetter(true, true, true), setdir{ 0, 0, 1 },
        inv_xscale(1 / xscale), inv_yscale(1 / yscale) {};
};

/**
 * @brief verifies that the acoustic-elastic coupling is exact for fields of a
 * certain power.
 *
 * Given the field on the interface as f(x,y) = (x/xscale)^{xpow}
 * (y/yscale)^{ypow} (or f multiplied by a constant vector {0,0,1} for elastic),
 * verifies that setting field on source_medium to f and calling the
 * compute_coupling routines produces the same acceleration field (on
 * target_medium) as if we computed the integral of the shape function times f
 * there.
 */
template <specfem::element::medium_tag target_medium,
          specfem::element::medium_tag source_medium, int NGLL, int pow_x,
          int pow_y>
void test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>
        &target_intersection_faces,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::
        host_mirror_type &h_target_intersection_faces,
    const Kokkos::View<type_real *[3]> &target_medium_normal_per_dof,
    std::integral_constant<int, NGLL>, std::integral_constant<int, pow_x>,
    std::integral_constant<int, pow_y>) {
  static_assert((target_medium == specfem::element::medium_tag::acoustic &&
                 source_medium == specfem::element::medium_tag::elastic) ||
                    (source_medium == specfem::element::medium_tag::acoustic &&
                     target_medium == specfem::element::medium_tag::elastic),
                "test_nonconforming_acoustic_elastic -- target_medium and "
                "source_medium must be acoustic and elastic (or vice versa)!");
  constexpr type_real reltol = 1e-5;
  constexpr type_real abstol = 1e-7;
  constexpr int fail_num_verbose = 5;

  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  const auto simfield = assembly.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();

  using target_initializer_type =
      std::conditional_t<target_medium ==
                             specfem::element::medium_tag::acoustic,
                         acoustic_field_initializer_pow<pow_x, pow_y>,
                         elastic_field_initializer_pow<pow_x, pow_y>>;
  using source_initializer_type =
      std::conditional_t<source_medium ==
                             specfem::element::medium_tag::acoustic,
                         acoustic_field_initializer_pow<pow_x, pow_y>,
                         elastic_field_initializer_pow<pow_x, pow_y>>;

  // target_initializer sets disp or accel, based on medium. Which is it?
  constexpr specfem::data_access::DataClassType
      target_initialized_component_dataclass =
          (target_medium == specfem::element::medium_tag::acoustic)
              ? specfem::data_access::DataClassType::acceleration
              : specfem::data_access::DataClassType::displacement;
  constexpr int ncomp_target =
      specfem::element::attributes<dimension_tag, target_medium>::components;
  static constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;

  // ================================================================================

  const type_real xscale =
      std::max(std::abs(assembly.mesh.xmax), std::abs(assembly.mesh.xmin));
  const type_real yscale =
      std::max(std::abs(assembly.mesh.ymax), std::abs(assembly.mesh.ymin));

  target_initializer_type target_initializer(xscale, yscale);
  source_initializer_type source_initializer(xscale, yscale);

  // set and integrate function on target side exclusively.
  specfem::test_fieldmanip::set_field_values<
      specfem::simulation::field_type::forward>(assembly, target_initializer);
  const auto target_set_fieldvals = specfem::test_fieldmanip::get_field_values<
      specfem::simulation::field_type::forward, dimension_tag, target_medium,
      target_initialized_component_dataclass>(assembly);

  const int nglob = target_set_fieldvals.extent(0);

  if constexpr (target_medium == specfem::element::medium_tag::acoustic) {
    // elastic set to constant direction. dot it with normal
    Kokkos::parallel_for(
        "target_set_fieldvals times normal", nglob,
        KOKKOS_LAMBDA(const int &iglob) {
          type_real dot = 0;
          for (int idim = 0; idim < ndim; idim++) {
            dot += source_initializer.setdir[idim] *
                   target_medium_normal_per_dof(iglob, idim);
          }
          target_set_fieldvals(iglob, 0) *= dot;
        });
  } else if constexpr (source_medium ==
                       specfem::element::medium_tag::acoustic) {
    // elastic set to constant direction. replace it it with proper normal
    type_real conormal[ndim];
    type_real normal_mag2 = 0;
    for (int idim = 0; idim < ndim; idim++) {
      normal_mag2 +=
          target_initializer.setdir[idim] * target_initializer.setdir[idim];
    }
    for (int idim = 0; idim < ndim; idim++) {
      conormal[idim] = target_initializer.setdir[idim] / normal_mag2;
    }

    Kokkos::parallel_for(
        "target_set_fieldvals times normal", nglob,
        KOKKOS_LAMBDA(const int &iglob) {
          type_real fieldval = 0;
          for (int idim = 0; idim < ndim; idim++) {
            fieldval += conormal[idim] * target_set_fieldvals(iglob, idim);
          }
          for (int idim = 0; idim < ndim; idim++) {
            target_set_fieldvals(iglob, idim) =
                target_medium_normal_per_dof(iglob, idim) * fieldval;
          }
        });
  }
  Kokkos::fence();

  const auto expected_accel_vals =
      specfem::nonconforming_test::kernel::integrate_against_shape_on_faces<
          target_medium>(assembly, target_intersection_faces,
                         h_target_intersection_faces, target_set_fieldvals);

  // ================================================================================
  // initialize source
  specfem::test_fieldmanip::set_field_values<
      specfem::simulation::field_type::forward>(assembly, source_initializer);

  // zero out target medium (since compute_coupling accumulates) and compute
  // through kernel
  specfem::test_fieldmanip::set_field_values<
      specfem::simulation::field_type::forward>(
      assembly,
      specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3, target_medium>(
          true, true, true) /* default point setter zeroes everyhing out. */);

  specfem::compute::impl::compute_coupling<
      NGLL, specfem::tags::Tags<dimension_tag,
                                specfem::simulation::field_type::forward,
                                target_medium>>(assembly);

  const auto computed_integrated_values =
      specfem::test_fieldmanip::get_field_values<
          specfem::simulation::field_type::forward, dimension_tag,
          target_medium,
          specfem::data_access::DataClassType::
              acceleration /* always accel: values from compute_coupling */>(
          assembly);
  // ================================================================================

  int iglob_fails = 0;

  const auto h_expected_accel_vals = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), expected_accel_vals);
  const auto h_target_set_fieldvals = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), target_set_fieldvals);
  const auto h_computed_integrated_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), computed_integrated_values);

  std::ostringstream failstream;
  type_real maxerr_rel = 0;
  type_real maxerr_abs = 0;
  for (int iglob = 0; iglob < nglob; iglob++) {
    // flip expected for opposite normal. (the [0,0,1] was static-constexpr'd)
    type_real expected[ncomp_target];
    type_real got[ncomp_target];
    type_real resid[ncomp_target];

    type_real expected_mag = 0;
    for (int icomp = 0; icomp < ncomp_target; icomp++) {
      expected[icomp] = h_expected_accel_vals(iglob, icomp);
      expected_mag += expected[icomp] * expected[icomp];
      got[icomp] = h_computed_integrated_values(iglob, icomp);
      resid[icomp] = got[icomp] - expected[icomp];
    }
    expected_mag = std::sqrt(expected_mag);

    type_real l2err = 0;
    for (int icomp = 0; icomp < ncomp_target; icomp++) {
      l2err += resid[icomp] * resid[icomp];
    }
    l2err = std::sqrt(l2err);

    const type_real tolparam = reltol * expected_mag + abstol;
    if (expected_mag > abstol) {
      maxerr_rel = std::max(l2err / expected_mag, maxerr_rel);
    }
    maxerr_abs = std::max(l2err, maxerr_abs);
    if (l2err > tolparam) {
      if (iglob_fails < fail_num_verbose) {

        // find local nodes that contribute to this DoF
        std::ostringstream collected_locals;
        for (int iface = 0; iface < target_intersection_faces.N; iface++) {
          const auto face = h_target_intersection_faces(iface);
          for (int ipoint = 0; ipoint < NGLL; ipoint++) {
            for (int jpoint = 0; jpoint < NGLL; jpoint++) {
              const auto index = face(ipoint, jpoint);
              const int iglob_of_point =
                  simfield.template get_iglob<false, target_medium>(index);
              if (iglob == iglob_of_point) {

                specfem::point::jacobian_matrix<
                    dimension_tag, true /*StoreJacobian*/, false /*UseSIMD*/>
                    jac;

                specfem::assembly::load_on_host(index, assembly.jacobian_matrix,
                                                jac);

                const auto normal = jac.compute_normal(index.face_type);
                type_real jac2d = std::sqrt(normal(0) * normal(0) + normal(1) +
                                            normal(1) + normal(2) * normal(2));
                type_real contrib = jac2d *
                                    assembly.mesh.h_weights(index.ipoint_i) *
                                    assembly.mesh.h_weights(index.ipoint_j);
                collected_locals
                    << "    (" << index.ispec << "," << index.ipoint_i << ","
                    << index.ipoint_j << ") field value * " << contrib << ": "
                    << "  jac(" << jac2d << ") * gll_weight("
                    << assembly.mesh.h_weights(index.ipoint_i)
                    << ") * gll_weight("
                    << assembly.mesh.h_weights(index.ipoint_j) << ")\n";
              }
            }
          }
        }
        failstream << "- iglob = " << iglob << ":\n";
        failstream << "    [ " << std::setw(15) << got[0];
        for (int icomp = 1; icomp < ncomp_target; icomp++) {
          failstream << ", " << std::setw(15) << got[icomp];
        }
        failstream << " ]";
        failstream << "\n != [ " << std::setw(15) << expected[0];
        for (int icomp = 1; icomp < ncomp_target; icomp++) {
          failstream << ", " << std::setw(15) << expected[icomp];
        }
        failstream << " ]";
        failstream << "\n        (rel err: " << l2err / expected_mag
                   << ")\n  local points:\n"
                   << collected_locals.str();
        failstream << "   field value: [ " << std::setw(15)
                   << h_target_set_fieldvals(iglob, 0);
        for (int icomp = 1; icomp < ncomp_target; icomp++) {
          failstream << ", " << std::setw(15)
                     << h_target_set_fieldvals(iglob, icomp);
        }
        failstream << " ]\n";
      }

      iglob_fails++;
    }
  }
  if (iglob_fails > 0) {
    FAIL() << meshname << std::endl
           << specfem::element::to_string(target_medium) << " <- "
           << specfem::element::to_string(source_medium) << ": x^{" << pow_x
           << "} y^{" << pow_y
           << "}\nFailed Degrees of Freedom: " << iglob_fails << " / " << nglob
           << "\n    Largest relative error: " << maxerr_rel
           << "\n    Largest absolute error: " << maxerr_abs
           << "\n Showing first " << fail_num_verbose << ":\n"
           << failstream.str();
  }
}

/**
 * @brief for each DoF, estimate the normal vector by averaging over all its
 * elements.
 */
template <specfem::element::medium_tag medium_tag>
Kokkos::View<type_real *[3]> get_target_medium_normal_per_dof(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const specfem::assembly::FaceView<
        Kokkos::DefaultExecutionSpace>::host_mirror_type &h_faces) {

  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr int ndim = specfem::element::dimension<dimension_tag>::dim;

  const auto field = assembly.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int nglob = field.template get_nglob<medium_tag>();

  Kokkos::View<type_real *[ndim]> norms("norms", nglob);
  Kokkos::View<type_real *[ndim]>::host_mirror_type h_norms =
      Kokkos::create_mirror_view(norms);
  Kokkos::deep_copy(h_norms, 0);

  // accumulate norms (larger elements (so larger jacs) are weighted more)
  for (int iface = 0; iface < h_faces.N; iface++) {
    const auto face = h_faces(iface);
    for (int ipoint = 0; ipoint < h_faces.n_points; ipoint++) {
      for (int jpoint = 0; jpoint < h_faces.n_points; jpoint++) {
        const auto index = face(ipoint, jpoint);
        const int iglob = field.template get_iglob<false, medium_tag>(index);
        specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3,
                                        true /*StoreJacobian*/,
                                        false /*UseSIMD*/>
            jac;

        specfem::assembly::load_on_host(index, assembly.jacobian_matrix, jac);
        const auto normal = jac.compute_normal(index.face_type);
        for (int idim = 0; idim < ndim; idim++) {
          h_norms(iglob, idim) += normal(idim);
        }
      }
    }
  }

  // normalize each
  for (int iglob = 0; iglob < nglob; iglob++) {
    type_real inv_norm_mag = 0;
    for (int idim = 0; idim < ndim; idim++) {
      inv_norm_mag += h_norms(iglob, idim) * h_norms(iglob, idim);
    }
    inv_norm_mag = 1 / std::sqrt(inv_norm_mag);
    for (int idim = 0; idim < ndim; idim++) {
      h_norms(iglob, idim) *= inv_norm_mag;
    }
  }

  Kokkos::deep_copy(norms, h_norms);
  return norms;
}

/**
 * @brief Expanded call (Is = 0,..., (MAXPOW+1)^2 - 1) to run tests for all
 * powers <= MAXPOW
 *
 * Runs test_nonconforming_acoustic_elastic() and
 * test_nonconforming_elastic_acoustic() for the grid
 *          0 <= pow_x, pow_y <= MAXPOW
 *
 * This should only be called by expand_test_pows(assembly).
 */
template <int NGLL, int MAXPOW, int... Is>
void expand_test_pows(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname, std::integer_sequence<int, Is...>) {

  const auto [acoustic_intersection_faces, h_acoustic_intersection_faces] =
      specfem::nonconforming_test::kernel::get_self_faces_on_intersection<
          specfem::element_coupling::interface_tag::acoustic_elastic>(assembly);
  const auto acoustic_norms =
      get_target_medium_normal_per_dof<specfem::element::medium_tag::acoustic>(
          assembly, h_acoustic_intersection_faces);
  (test_nonconforming_acoustic_elastic<specfem::element::medium_tag::acoustic,
                                       specfem::element::medium_tag::elastic>(
       assembly, meshname, acoustic_intersection_faces,
       h_acoustic_intersection_faces, acoustic_norms,
       std::integral_constant<int, 5>(),
       std::integral_constant<int, Is % (MAXPOW + 1) /*pow_x*/>(),
       std::integral_constant<int, Is / (MAXPOW + 1) /*pow y*/>()),
   ...);

  const auto [elastic_intersection_faces, h_elastic_intersection_faces] =
      specfem::nonconforming_test::kernel::get_coupled_faces_on_intersection<
          specfem::element_coupling::interface_tag::acoustic_elastic>(assembly);
  const auto elastic_norms =
      get_target_medium_normal_per_dof<specfem::element::medium_tag::elastic>(
          assembly, h_elastic_intersection_faces);
  (test_nonconforming_acoustic_elastic<specfem::element::medium_tag::elastic,
                                       specfem::element::medium_tag::acoustic>(
       assembly, meshname, elastic_intersection_faces,
       h_elastic_intersection_faces, elastic_norms,
       std::integral_constant<int, 5>(),
       std::integral_constant<int, Is % (MAXPOW + 1) /*pow_x*/>(),
       std::integral_constant<int, Is / (MAXPOW + 1) /*pow y*/>()),
   ...);
}

/**
 * @brief Runs test on functions f(x,y,z) = x^{pow_x} y^{pow_y} up to MAXPOW
 *
 * Runs test_nonconforming_acoustic_elastic() and
 * test_nonconforming_elastic_acoustic() for the grid
 *          0 <= pow_x, pow_y <= MAXPOW
 */
template <int NGLL, int MAXPOW>
void expand_test_pows(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname) {
  expand_test_pows<NGLL, MAXPOW>(
      assembly, meshname,
      std::make_integer_sequence<int, (MAXPOW + 1) * (MAXPOW + 1)>());
}

void specfem::nonconforming_test::kernel::test_nonconforming_acoustic_elastic(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname) {
  const int ngll = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;

  if (ngll == 5) {
    expand_test_pows<5, 3>(assembly, meshname);
  } else if (ngll == 8) {
    expand_test_pows<8, 4>(assembly, meshname);
  } else {
    std::ostringstream oss;
    oss << "specfem::nonconforming_test::kernel::test_nonconforming_acoustic_"
           "elastic NGLL = "
        << ngll << " not configured!";
    throw std::runtime_error(oss.str());
  }
}
