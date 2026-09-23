#include "../acoustic_elastic.hpp"
#include "Kokkos_Macros.hpp"
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

template <int N> struct sum_reduction_array {
  type_real arr[N];

  KOKKOS_INLINE_FUNCTION sum_reduction_array() {
    for (int i = 0; i < N; i++) {
      arr[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION type_real &operator[](const int &i) { return arr[i]; }
  KOKKOS_INLINE_FUNCTION const type_real &operator[](const int &i) const {
    return arr[i];
  }
  KOKKOS_INLINE_FUNCTION sum_reduction_array &
  operator+=(const sum_reduction_array &other) {
    for (int i = 0; i < N; i++) {
      arr[i] += other[i];
    }
    return (*this);
  }
};

/**
 * @brief verifies that the conjugate natural acoustic-elastic coupling
 *
 * The conjugate kernel computes the coupling integral on the coupling side.
 * We can verify correctness by taking linear combinations of test functions
 * that are exact for both coupling-sided and self-sided integration (i.e.
 * powers). pow_x and pow_y determine the source field (the linear combinations
 * of columns are chosen), while this function iterates over linear combinations
 * of rows (varies test functions).
 */
template <specfem::element::medium_tag target_medium,
          specfem::element::medium_tag source_medium, int NGLL, int pow_x,
          int pow_y>
void test_nonconforming_acoustic_elastic_conj(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>
        &target_intersection_faces,
    const specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>::
        host_mirror_type &h_target_intersection_faces,
    const Kokkos::View<type_real *[3]> &target_medium_normal_per_dof,
    const int &max_pow_test, std::integral_constant<int, NGLL>,
    std::integral_constant<int, pow_x>, std::integral_constant<int, pow_y>) {
  static_assert((target_medium == specfem::element::medium_tag::acoustic &&
                 source_medium == specfem::element::medium_tag::elastic) ||
                    (source_medium == specfem::element::medium_tag::acoustic &&
                     target_medium == specfem::element::medium_tag::elastic),
                "test_nonconforming_acoustic_elastic -- target_medium and "
                "source_medium must be acoustic and elastic (or vice versa)!");

  constexpr type_real reltol = 1e-5;
  constexpr type_real abstol = 1e-7;
  constexpr int fail_num_verbose = 5;

  using target_initializer_type = std::conditional_t<
      target_medium == specfem::element::medium_tag::acoustic,
      specfem::nonconforming_test::kernel::acoustic_field_initializer_pow<
          pow_x, pow_y>,
      specfem::nonconforming_test::kernel::elastic_field_initializer_pow<
          pow_x, pow_y>>;
  using source_initializer_type = std::conditional_t<
      source_medium == specfem::element::medium_tag::acoustic,
      specfem::nonconforming_test::kernel::acoustic_field_initializer_pow<
          pow_x, pow_y>,
      specfem::nonconforming_test::kernel::elastic_field_initializer_pow<
          pow_x, pow_y>>;
  constexpr auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr int ncomp_target =
      specfem::element::attributes<dimension_tag, target_medium>::components;

  const type_real xscale =
      std::max(std::abs(assembly.mesh.xmax), std::abs(assembly.mesh.xmin));
  const type_real yscale =
      std::max(std::abs(assembly.mesh.ymax), std::abs(assembly.mesh.ymin));
  const type_real inv_xscale = 1 / xscale;
  const type_real inv_yscale = 1 / yscale;

  target_initializer_type target_initializer(xscale, yscale);
  source_initializer_type source_initializer(xscale, yscale);

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
          true, true, true) /* default point setter zeroes everything out. */);
  specfem::compute::impl::compute_coupling<
      NGLL, specfem::tags::Tags<dimension_tag,
                                specfem::simulation::field_type::forward,
                                target_medium>>(assembly);
  const auto integrated_base_values =
      specfem::test_fieldmanip::get_field_values<
          specfem::simulation::field_type::forward, dimension_tag,
          target_medium,
          specfem::data_access::DataClassType::
              acceleration /* always accel: values from compute_coupling */>(
          assembly);

  // zero out target medium (since compute_coupling accumulates) and compute
  // through kernel
  specfem::test_fieldmanip::set_field_values<
      specfem::simulation::field_type::forward>(
      assembly,
      specfem::test_fieldmanip::PointSetter<
          specfem::element::dimension_tag::dim3, target_medium>(
          true, true, true) /* default point setter zeroes everything out. */);

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<dimension_tag>{} *
          CONNECTION_SET(nonconforming) *
          INTERFACE_SET(elastic_acoustic, acoustic_elastic) *
          BOUNDARY_SET(none, acoustic_free_surface, stacey,
                       composite_stacey_dirichlet) *
          FLUX_SCHEME_SET(natural),
      [&]<typename ElementTags>() {
        constexpr auto self_medium = specfem::element_coupling::attributes<
            ElementTags::dimension_tag,
            ElementTags::interface_tag>::self_medium();
        if constexpr (self_medium == target_medium) {
          specfem::compute::impl::
              compute_coupling_conjugate_integral_nonconforming<
                  NGLL,
                  specfem::tags::Tags<
                      dimension_tag, ElementTags::connection_tag,
                      specfem::simulation::field_type::forward,
                      ElementTags::interface_tag, ElementTags::boundary_tag,
                      ElementTags::flux_scheme_tag>>(assembly);
        }
      });
  const auto integrated_conjugate_values =
      specfem::test_fieldmanip::get_field_values<
          specfem::simulation::field_type::forward, dimension_tag,
          target_medium,
          specfem::data_access::DataClassType::
              acceleration /* always accel: values from compute_coupling */>(
          assembly);

  Kokkos::View<specfem::point::global_coordinates<dimension_tag> *> dof_coords =
      specfem::test_fieldmanip::get_coords_per_dof<
          specfem::simulation::field_type::forward, target_medium>(assembly);

  const int nglob_target = dof_coords.extent(0);

  int num_fails = 0;
  // error: vec2norm(integral(shape_poly . coupling_stress . field_poly))
  // for relative, we divide by:
  //        integral(vec2norm(shape_poly . coupling_stress . field_poly))
  type_real maxerr_rel = 0;
  type_real maxerr_abs = 0;
  const int num_poly_checks = (max_pow_test + 1) * (max_pow_test + 1);
  std::ostringstream failstream;

  for (int test_pow_x = 0; test_pow_x <= max_pow_test; test_pow_x++) {
    for (int test_pow_y = 0; test_pow_y <= max_pow_test; test_pow_y++) {

      sum_reduction_array<ncomp_target + 1> result_base; //+1 for magnitude
                                                         //(scale for error)
      sum_reduction_array<ncomp_target> result_conjugate;

      Kokkos::parallel_reduce(
          "reduce base", nglob_target,
          KOKKOS_LAMBDA(const int &iglob,
                        sum_reduction_array<ncomp_target + 1> &lsum) {
            const type_real location_factor =
                std::pow(dof_coords(iglob).x * inv_xscale, test_pow_x) *
                std::pow(dof_coords(iglob).y * inv_yscale, test_pow_y);
            type_real mag2 = 0;
            for (int i = 0; i < ncomp_target; i++) {
              type_real addend =
                  integrated_base_values(iglob, i) * location_factor;
              lsum[i] += addend;
              mag2 += addend * addend;
            }
            lsum[ncomp_target] += std::sqrt(mag2);
          },
          result_base);
      Kokkos::parallel_reduce(
          "reduce conjugate", nglob_target,
          KOKKOS_LAMBDA(const int &iglob,
                        sum_reduction_array<ncomp_target> &lsum) {
            const type_real location_factor =
                std::pow(dof_coords(iglob).x * inv_xscale, test_pow_x) *
                std::pow(dof_coords(iglob).y * inv_yscale, test_pow_y);
            for (int i = 0; i < ncomp_target; i++) {
              lsum[i] +=
                  integrated_conjugate_values(iglob, i) * location_factor;
            }
          },
          result_conjugate);

      type_real err = 0;
      const type_real base_mag = result_base[ncomp_target];
      for (int icomp = 0; icomp < ncomp_target; icomp++) {
        err += (result_base[icomp] - result_conjugate[icomp]) *
               (result_base[icomp] - result_conjugate[icomp]);
      }
      err = std::sqrt(err);
      maxerr_abs = std::max(err, maxerr_abs);
      if (base_mag > abstol) {
        maxerr_rel = std::max(err / base_mag, maxerr_rel);
      }

      if (err > base_mag * reltol + abstol) {

        if (num_fails < fail_num_verbose) {
          failstream << "- x^{" << test_pow_x << "} y^{" << test_pow_y << "}\n";
          failstream << "    [ " << std::setw(15) << result_conjugate[0];
          for (int icomp = 1; icomp < ncomp_target; icomp++) {
            failstream << ", " << std::setw(15) << result_conjugate[icomp];
          }
          failstream << " ]";
          failstream << "\n != [ " << std::setw(15) << result_base[0];
          for (int icomp = 1; icomp < ncomp_target; icomp++) {
            failstream << ", " << std::setw(15) << result_base[icomp];
          }
          failstream << " ]\n        (rel err: " << err / base_mag
                     << ")\n        (rel-to : " << base_mag << ")\n";
        }
        num_fails++;
      }
    }
  }

  if (num_fails > 0) {
    FAIL() << meshname << std::endl
           << specfem::element::to_string(target_medium) << " <- "
           << specfem::element::to_string(source_medium) << ": x^{" << pow_x
           << "} y^{" << pow_y << "} conjugate comparison"
           << "\nFailed polynomial test functions: " << num_fails << " / "
           << num_poly_checks
           << "\n          Largest relative error: " << maxerr_rel
           << "\n          Largest absolute error: " << maxerr_abs
           << "\n (relative error is relative to integral of integrand "
              "pointwise l2-norm)"
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
void expand_test_pows_conj(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname, std::integer_sequence<int, Is...>) {

  const auto [acoustic_intersection_faces, h_acoustic_intersection_faces] =
      specfem::nonconforming_test::kernel::get_self_faces_on_intersection<
          specfem::element_coupling::interface_tag::acoustic_elastic>(assembly);
  const auto acoustic_norms =
      get_target_medium_normal_per_dof<specfem::element::medium_tag::acoustic>(
          assembly, h_acoustic_intersection_faces);
  (test_nonconforming_acoustic_elastic_conj<
       specfem::element::medium_tag::acoustic,
       specfem::element::medium_tag::elastic>(
       assembly, meshname, acoustic_intersection_faces,
       h_acoustic_intersection_faces, acoustic_norms, MAXPOW,
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
  (test_nonconforming_acoustic_elastic_conj<
       specfem::element::medium_tag::elastic,
       specfem::element::medium_tag::acoustic>(
       assembly, meshname, elastic_intersection_faces,
       h_elastic_intersection_faces, elastic_norms, MAXPOW,
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
void expand_test_pows_conj(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::string &meshname) {
  expand_test_pows_conj<NGLL, MAXPOW>(
      assembly, meshname,
      std::make_integer_sequence<int, (MAXPOW + 1) * (MAXPOW + 1)>());
}

void specfem::nonconforming_test::kernel::
    test_nonconforming_acoustic_elastic_conjugate(
        const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
            &assembly,
        const std::string &meshname) {
  const int ngll = assembly.mesh.specfem::assembly::mesh_impl::points<
      specfem::element::dimension_tag::dim3>::ngllz;

  if (ngll == 5) {
    expand_test_pows_conj<5, 3>(assembly, meshname);
  } else if (ngll == 8) {
    expand_test_pows_conj<8, 4>(assembly, meshname);
  } else {
    std::ostringstream oss;
    oss << "specfem::nonconforming_test::kernel::test_nonconforming_acoustic_"
           "elastic NGLL = "
        << ngll << " not configured!";
    throw std::runtime_error(oss.str());
  }
}
