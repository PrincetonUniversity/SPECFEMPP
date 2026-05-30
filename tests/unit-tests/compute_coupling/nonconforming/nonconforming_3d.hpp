#pragma once

#include "specfem/algorithms/transfer_interpolate.hpp"

#include "specfem/data_access/accessor.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/element_coupling/accessor.hpp"
#include "specfem/medium_physics.hpp"

#include "specfem/datatype/accessor_type.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element_coupling/accessor.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/element_coupling/tags.hpp"
#include "specfem/point/acceleration.hpp"
#include "specfem/point/face_index.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"

// for interpolator
#include "specfem/quadrature/gll.hpp"

#include "utilities/include/fixture/accessor.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/face_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>

namespace specfem::compute_coupling_test {

/**
 * @brief Test index type for chunk face operations.
 */
class ChunkFaceIndex {
public:
  static constexpr auto accessor_type =
      specfem::datatype::AccessorType::chunk_face;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  /**
   * @brief Get Kokkos team member index.
   * @return Reference to Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Construct chunk face index.
   * @param nfaces Number of faces in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkFaceIndex(const int nfaces, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), chunk_size(nfaces) {}

  const int chunk_size; ///< Number of faces in the chunk
private:
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
};

namespace compute_coupling {

template <typename T, typename = void>
struct is_analytical : std::false_type {};

template <typename T>
struct is_analytical<
    T,
    std::enable_if_t<T::FunctionInitializer::is_from_analytical_function, void>>
    : std::true_type {};

template <typename T, typename = void> struct is_stack : std::false_type {};

template <typename T>
struct is_stack<
    T, std::enable_if_t<T::FunctionInitializer::is_stacked_initializer, void>>
    : std::true_type {};

template <specfem::element::dimension_tag dimension_tag,
          specfem::element_coupling::interface_tag interface_tag>
constexpr int ncomp_self_from_interface_tag = specfem::element::attributes<
    dimension_tag,
    specfem::element_coupling::attributes<
        dimension_tag, interface_tag>::self_medium()>::components;

// analytically known function (just evaluate function at coords)
template <specfem::element::dimension_tag dimension_tag,
          specfem::element_coupling::interface_tag interface_tag,
          typename TransferCoordinates, typename SelfNormalFunction,
          typename FaceFunction>
std::enable_if_t<
    is_analytical<FaceFunction>::value,
    std::vector<std::array<
        std::array<std::array<type_real, ncomp_self_from_interface_tag<
                                             dimension_tag, interface_tag>>,
                   TransferCoordinates::nquad_element>,
        TransferCoordinates::nquad_element>>>
expected_solution(const TransferCoordinates &transfer_coordinates,
                  const SelfNormalFunction &self_normal,
                  const FaceFunction &face_function) {

  constexpr int ncomp_self =
      ncomp_self_from_interface_tag<dimension_tag, interface_tag>;
  std::vector<std::array<
      std::array<std::array<type_real, ncomp_self_from_interface_tag<
                                           dimension_tag, interface_tag>>,
                 TransferCoordinates::nquad_element>,
      TransferCoordinates::nquad_element>>
      result(FaceFunction::stack_size);
  for (int istack = 0; istack < FaceFunction::stack_size; ++istack) {
    for (int ipoint1 = 0; ipoint1 < TransferCoordinates::nquad_element;
         ++ipoint1) {
      for (int ipoint2 = 0; ipoint2 < TransferCoordinates::nquad_element;
           ++ipoint2) {
        type_real coord1 = transfer_coordinates(istack, ipoint1, ipoint2, 0);
        type_real coord2 = transfer_coordinates(istack, ipoint1, ipoint2, 1);
        if (std::isnan(coord1)) {
          for (int icomp = 0; icomp < ncomp_self; ++icomp) {
            result[istack][ipoint1][ipoint2][icomp] = 0;
          }
        } else {
          const auto point_eval = FaceFunction::FunctionInitializer::
              AnalyticalFunctionType::evaluate(coord1, coord2);

          if constexpr (interface_tag == specfem::element_coupling::
                                             interface_tag::acoustic_elastic) {

            result[istack][ipoint1][ipoint2][0] = 0;

            for (int icomp = 0; icomp < FaceFunction::num_components; ++icomp) {
              result[istack][ipoint1][ipoint2][0] +=
                  point_eval[icomp] *
                  self_normal(istack, ipoint1, ipoint2, icomp);
            }
          } else if constexpr (interface_tag ==
                               specfem::element_coupling::interface_tag::
                                   elastic_acoustic) {
            for (int icomp = 0; icomp < ncomp_self; ++icomp) {
              result[istack][ipoint1][ipoint2][icomp] =
                  point_eval[0] * self_normal(istack, ipoint1, ipoint2, icomp);
            }
          }
        }
      }
    }
  }
  return result;
}

// stacked functions -- split apart
template <specfem::element::dimension_tag dimension_tag,
          specfem::element_coupling::interface_tag interface_tag,
          typename TransferCoordinates, typename SelfNormalFunction,
          typename... FaceFunctionsInStack>
std::vector<std::array<
    std::array<std::array<type_real, ncomp_self_from_interface_tag<
                                         dimension_tag, interface_tag>>,
               TransferCoordinates::nquad_element>,
    TransferCoordinates::nquad_element>>
expected_solution(const TransferCoordinates &transfer_coordinates,
                  const SelfNormalFunction &self_normal,
                  const specfem::test_fixture::FaceFunction3D<
                      specfem::test_fixture::FaceFunctionInitializer3D::Stack<
                          FaceFunctionsInStack...>> &face_function) {
  using FaceFunction = std::decay_t<decltype(face_function)>;
  std::vector<std::array<
      std::array<std::array<type_real, ncomp_self_from_interface_tag<
                                           dimension_tag, interface_tag>>,
                 TransferCoordinates::nquad_element>,
      TransferCoordinates::nquad_element>>
      result(FaceFunction::stack_size);
  auto it = result.begin();
  (
      [&]() {
        // override transfer_coordinates with FaceFunction stack offset
        struct slice_coords : TransferCoordinates {
          slice_coords(const TransferCoordinates &ref, const int &offset)
              : TransferCoordinates(ref), offset(offset) {}
          const int offset;
          const type_real &operator()(const int i, const int j, const int k,
                                      const int ell) const {
            return (*this).TransferCoordinates::operator()(i + offset, j, k,
                                                           ell);
          }
        };
        struct slice_normals : SelfNormalFunction {
          slice_normals(const SelfNormalFunction &ref, const int &offset)
              : SelfNormalFunction(ref), offset(offset) {}
          const int offset;
          const type_real &operator()(const int i, const int j, const int k,
                                      const int ell) const {
            return (*this).SelfNormalFunction::operator()(i + offset, j, k,
                                                          ell);
          }
        };

        const auto sub = expected_solution(
            slice_coords(transfer_coordinates, it - result.begin()),
            slice_normals(self_normal, it - result.begin()),
            specfem::test_fixture::FaceFunction3D(FaceFunctionsInStack()));
        std::copy(sub.begin(), sub.end(), it);
        it += FaceFunctionsInStack::stack_size;
      }(),
      ...);

  return result;
}

template <typename TransferCoordinates, typename FaceFunction,
          typename SelfNormalFunction>
void execute(const TransferCoordinates &transfer_coordinates,
             const FaceFunction &face_function,
             const SelfNormalFunction &self_normal_function,
             const std::string &execution_description = {}) {

  // ======= dummy declarations
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  constexpr int chunk_size =
      specfem::parallel_configuration::default_chunk_face_config<
          specfem::element::dimension_tag::dim3,
          Kokkos::DefaultExecutionSpace>::chunk_size;

  /** Test dimension (3D) */
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  /** Interface type (dummy for testing) */
  constexpr static auto interface_tag =
      specfem::element_coupling::interface_tag::acoustic_elastic;
  /** Boundary type (dummy for testing) */
  constexpr static auto boundary_tag = specfem::element::boundary_tag::none;
  /** Flux scheme tag (dummy for testing) */
  constexpr static auto flux_scheme_tag =
      specfem::element_coupling::flux_scheme_tag::natural;

  // ======= consequential declarations
  constexpr int ngll_coupled_face = FaceFunction::nquad_element;
  constexpr int ngll_self = TransferCoordinates::nquad_element;
  constexpr int stack_size = FaceFunction::stack_size;
  constexpr int league_size = (stack_size + chunk_size - 1) / chunk_size;

  using interface_attrib =
      specfem::element_coupling::attributes<dimension_tag, interface_tag>;
  constexpr auto self_medium = interface_attrib::self_medium();
  constexpr auto coupled_medium = interface_attrib::coupled_medium();
  constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;
  constexpr int ncomp_coupled =
      specfem::element::attributes<dimension_tag, coupled_medium>::components;

  static_assert(ncomp_coupled == FaceFunction::num_components,
                "coupled FaceFunction does not have the correct number of "
                "components for this interface tag!");

  constexpr specfem::data_access::DataClassType coupled_field_dataclass =
      (interface_tag ==
       specfem::element_coupling::interface_tag::elastic_acoustic)
          ? specfem::data_access::DataClassType::acceleration
          : specfem::data_access::DataClassType::displacement;

  // ======= Declare Kernel Container Types
  // using ContainerTransferCoordinates = specfem::test_fixture::EndowAccessor<
  //     specfem::element_coupling::accessor::coupling_terms_pack<
  //         dimension_tag, interface_tag, boundary_tag, flux_scheme_tag,
  //         chunk_size, ngll_self, ngll_self, Kokkos::DefaultExecutionSpace,
  //         Kokkos::MemoryTraits<>, Kokkos::LayoutRight>,
  //     specfem::tags::Tags<dimension_tag, interface_tag>>;

  using ContainerFunctionCoupled = specfem::test_fixture::EndowAccessor<
      specfem::test_fixture::EndowAccessor<
          specfem::datatype::VectorChunkFaceViewType<
              type_real, dimension_tag, chunk_size, ngll_coupled_face,
              FaceFunction::num_components, false /*UseSIMD*/,
              Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<>,
              Kokkos::LayoutRight>,
          specfem::tags::Tags<dimension_tag, coupled_medium>>,
      specfem::data_access::Accessor<
          specfem::datatype::AccessorType::chunk_face, coupled_field_dataclass,
          dimension_tag, false /*UseSIMD*/>>;

  static_assert(ContainerFunctionCoupled::accessor_type ==
                specfem::datatype::AccessorType::chunk_face);

  using SelfFieldType = specfem::point::acceleration<
      specfem::tags::Tags<dimension_tag, self_medium, false /*UseSIMD*/>>;

  using ResultViewType =
      Kokkos::View<type_real[stack_size][ngll_self][ngll_self][ncomp_self],
                   memory_space, Kokkos::MemoryTraits<>>;

  // ======= Generate GLL knots for interpolator

  const auto &gll_struct =
      specfem::quadrature::gll::gll(0, 0, ngll_coupled_face);
  specfem::algorithms::KnotInterpolator interpolator(gll_struct.get_xi());

  using ContainerTransferCoordinates = specfem::test_fixture::EndowAccessor<
      specfem::chunk_face::NonconformingAccessorPack<
          specfem::chunk_face::transfer_coupled_coordinates<
              dimension_tag, chunk_size, ngll_self, interface_tag, boundary_tag,
              flux_scheme_tag, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<>, Kokkos::LayoutRight>,
          specfem::chunk_face::intersection_normal<
              dimension_tag, interface_tag, boundary_tag, flux_scheme_tag,
              chunk_size, ngll_self, Kokkos::DefaultExecutionSpace,
              Kokkos::MemoryTraits<>, Kokkos::LayoutRight>,
          specfem::test_fixture::EndowAccessor<
              decltype(interpolator),
              specfem::tags::Tags<dimension_tag, interface_tag, boundary_tag,
                                  flux_scheme_tag>>>,
      specfem::tags::Tags<dimension_tag, interface_tag>>;

  // ======= Run Kernel
  const auto function_view =
      face_function.template get_chunkwise_view<chunk_size>();
  const auto normal_view =
      self_normal_function.template get_chunkwise_view<chunk_size>();
  const auto transfer_coord_view =
      transfer_coordinates.template get_chunkwise_view<chunk_size>();
  ResultViewType result_view("result_view");

  Kokkos::parallel_for(
      "transfer_interpolate_test",
      Kokkos::TeamPolicy<>(league_size, Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int ichunk = team_member.league_rank();
        const int this_chunk_start = ichunk * chunk_size;
        const int this_chunk_size =
            std::min(chunk_size, stack_size - this_chunk_start);

        const ContainerTransferCoordinates TF(
            Kokkos::subview(transfer_coord_view, ichunk, Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            Kokkos::subview(normal_view, ichunk, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL()),
            interpolator.xi_view);
        const ContainerFunctionCoupled F(
            Kokkos::subview(function_view, ichunk, Kokkos::ALL(), Kokkos::ALL(),
                            Kokkos::ALL(), Kokkos::ALL()));

        specfem::execution::for_each_level(
            specfem::execution::TeamThreadMDRangeIterator(
                team_member, this_chunk_size, ngll_self, ngll_self),
            [&](const auto &index) {
              specfem::point::face_index<dimension_tag> mocked_point_index(
                  0, index(0), index(1), index(2), 0, 0, 0,
                  specfem::mesh_entity::dim3::type::back);

              SelfFieldType self_field;
              specfem::medium_physics::compute_coupling(mocked_point_index, TF,
                                                        F, self_field);

              for (int icomp = 0; icomp < ncomp_self; ++icomp) {
                result_view(this_chunk_start + index(0), index(1), index(2),
                            icomp) = self_field(icomp);
              }
            });
      });

  Kokkos::fence();

  auto expected = expected_solution<dimension_tag, interface_tag>(
      transfer_coordinates, self_normal_function, face_function);
  auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);

  std::ostringstream fail_log;
  int num_fails = 0;

  for (int istack = 0; istack < stack_size; istack++) {
    std::ostringstream index_fail_log;
    int num_fails_at_index = 0;

    for (int ipoint_axis1 = 0; ipoint_axis1 < ngll_self; ipoint_axis1++) {
      for (int ipoint_axis2 = 0; ipoint_axis2 < ngll_self; ipoint_axis2++) {
        for (int icomp = 0; icomp < ncomp_self; icomp++) {
          if (!specfem::utilities::is_close(
                  result_host(istack, ipoint_axis1, ipoint_axis2, icomp),
                  expected[istack][ipoint_axis1][ipoint_axis2][icomp])) {
            num_fails++;
            num_fails_at_index++;
            index_fail_log
                << "  - face point (" << ipoint_axis1 << ", " << ipoint_axis2
                << ") @ local coordinates ("
                << transfer_coordinates(istack, ipoint_axis1, ipoint_axis2, 0)
                << ", "
                << transfer_coordinates(istack, ipoint_axis1, ipoint_axis2, 1)
                << "), component " << icomp << "\n     expected: "
                << expected[istack][ipoint_axis1][ipoint_axis2][icomp]
                << "\n          got: "
                << result_host(istack, ipoint_axis1, ipoint_axis2, icomp)
                << std::endl;
          }
        }
      }
    }

    if (num_fails_at_index > 0) {
      int ichunk = istack / chunk_size;
      fail_log << num_fails_at_index << " fails at face " << istack
               << "(local index " << istack - (ichunk * chunk_size)
               << " of chunk " << ichunk << ")\n"
               << index_fail_log.str();
    }
  }

  if (num_fails > 0) {
    std::ostringstream oss;
    oss << "============================================\n"
        << "transfer_interpolate failed\n"
        << "-- Transfer function --\n"
        << TransferCoordinates::description() << std::endl
        << "-- Edge Function --\n"
        << FaceFunction::description() << std::endl;
    if (!execution_description.empty()) {
      oss << "-- Execution Description --\n"
          << execution_description << std::endl;
    }
    oss << "============================================\n" << fail_log.str();
    ADD_FAILURE() << oss.str();
  }
}
} // namespace compute_coupling

} // namespace specfem::compute_coupling_test

// placeholder (?) struct for containing quadrature points for the test.
template <int StackSize, int NGLL> struct NoisyFaceQuadraturePoints3D {

  static constexpr int stack_size = StackSize;
  static constexpr int num_components = 2;
  static constexpr int nquad_element = NGLL;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

private:
  using ArrayType =
      std::array<std::array<std::array<std::array<type_real, num_components>,
                                       nquad_element>,
                            nquad_element>,
                 stack_size>;
  using FieldView = Kokkos::View<
      type_real[stack_size][nquad_element][nquad_element][num_components],
      memory_space>;
  using ResizedFieldView =
      Kokkos::View<type_real *[nquad_element][nquad_element][num_components],
                   memory_space>;
  ArrayType _field;

public:
  /**
   * @brief Builds quadrature points uniformly spaced on [xmin, xmax] x [ymin,
   * ymax], offset by noise of size noise_size * (max-min)/(NGLL-1) in each
   * dimension. If any of these points go beyond [-1,1] x [-1,1], they are NAN'd
   * out.
   *
   */
  NoisyFaceQuadraturePoints3D(const type_real &xmin, const type_real &xmax,
                              const type_real &ymin, const type_real &ymax,
                              const type_real noise_size = 0.3, int seed = 0) {
    type_real scale_x = (xmax - xmin) / (nquad_element - 1);
    type_real scale_y = (ymax - ymin) / (nquad_element - 1);
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<type_real> noise(-0.5 * noise_size,
                                                    0.5 * noise_size);

    for (size_t j = 0; j < nquad_element; ++j) {
      const type_real x0 = xmin + scale_x * j;
      for (size_t ell = 0; ell < nquad_element; ++ell) {
        const type_real y0 = ymin + scale_y * ell;
        for (size_t i = 0; i < stack_size; ++i) {
          type_real x = x0 + noise(rng) * scale_x;
          type_real y = y0 + noise(rng) * scale_y;

          if (x < -1 || x > 1 || y < -1 || y > 1) {
            x = NAN;
            y = NAN;
          }

          _field[i][j][ell][0] = x;
          _field[i][j][ell][1] = y;
        }
      }
    }
  }

  /**
   * @brief Get Kokkos view of field data.
   * @return Kokkos view for device access
   */
  FieldView get_view() const {
    FieldView view("field_view");
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < stack_size; ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i, j, ell, k) = (*this)(i, j, ell, k);
          }
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  /**
   * @brief Get Kokkos view of field data, set to different size.
   * @return Kokkos view for device access
   */
  ResizedFieldView get_resized_view(const int &stack_size) const {
    ResizedFieldView view("field_view", stack_size);
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < std::min(stack_size, this->stack_size); ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i, j, ell, k) = (*this)(i, j, ell, k);
          }
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  template <int chunk_size>
  Kokkos::View<
      type_real *[chunk_size][nquad_element][nquad_element][num_components],
      Kokkos::LayoutRight, memory_space>
  get_chunkwise_view(int stack_size = -1) const {
    if (stack_size < 0) {
      stack_size = this->stack_size;
    }
    const int num_stacks = (stack_size + chunk_size - 1) / chunk_size;
    Kokkos::View<
        type_real *[chunk_size][nquad_element][nquad_element][num_components],
        Kokkos::LayoutRight, memory_space>
        view("field_view", num_stacks);
    auto host_view =
        Kokkos::create_mirror_view(view); // Create host mirror to copy data
    for (size_t i = 0; i < std::min(stack_size, this->stack_size); ++i) {
      for (size_t j = 0; j < nquad_element; ++j) {
        for (size_t ell = 0; ell < nquad_element; ++ell) {
          for (size_t k = 0; k < num_components; ++k) {
            host_view(i / chunk_size, i % chunk_size, j, ell, k) =
                (*this)(i, j, ell, k);
          }
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  static std::string description() {
    return "NoisyFaceQuadraturePoints3D";
  }
  static std::string initializer_name() { return "<runtime-initialized>"; }

  const type_real &operator()(const int i, const int j, const int ell,
                              const int k) const {
    return _field[i][j][ell][k];
  }
};

template <typename QuadratureRuleType, int powerX, int powerY>
struct PowerGridStack {

private:
  template <int sliceX, int... powsY>
  static specfem::test_fixture::FaceFunctionInitializer3D::Stack<
      specfem::test_fixture::FaceFunctionInitializer3D::FromAnalyticalFunction<
          specfem::test_fixture::AnalyticalFunctionType::Power2D<sliceX, powsY>,
          QuadratureRuleType>...>
  PowerYStack(std::integer_sequence<int, powsY...>) {}

  template <int... powsX>
  static specfem::test_fixture::FaceFunctionInitializer3D::Stack<
      decltype(PowerYStack<powsX>(
          std::make_integer_sequence<int, powerY>()))...>
  PowerXStack(std::integer_sequence<int, powsX...>) {}

public:
  using type = decltype(PowerXStack(std::make_integer_sequence<int, powerX>()));
};
