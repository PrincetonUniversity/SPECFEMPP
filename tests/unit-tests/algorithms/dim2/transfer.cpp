
#include <array>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "algorithms/transfer.hpp"
#include "datatypes/chunk_edge_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/data_access.hpp"
#include "utilities/interface.hpp"

#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"

namespace specfem::algorithms_test {

/**
 * @brief Test index type for chunk edge operations.
 */
class ChunkEdgeIndex {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
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
   * @brief Construct chunk edge index.
   * @param nedges Number of edges in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;                  ///< Number of edges in the chunk
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
};

/** Test dimension (2D) */
constexpr static auto dimension_tag = specfem::dimension::type::dim2;
/** Interface type (dummy for testing) */
constexpr static auto interface_tag =
    specfem::interface::interface_tag::acoustic_elastic;
/** Boundary type (dummy for testing) */
constexpr static auto boundary_tag = specfem::element::boundary_tag::none;
constexpr size_t n_quad_element = 5; /**< Quadrature points per element */
constexpr size_t n_quad_intersection =
    5;                             /**< Quadrature points per intersection */
constexpr size_t n_components = 1; /**< Field components */
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

/**
 * @brief Transfer function initializers for testing.
 */
namespace TransferFunctionInitializer2D {
/** Zero transfer function initializer */
struct Zero {};
} // namespace TransferFunctionInitializer2D

/**
 * @brief Function initializers for testing.
 */
namespace EdgeFunctionInitializer2D {
/** Uniform field initializer */
struct Uniform {};
} // namespace EdgeFunctionInitializer2D

/**
 * @brief Initialize transfer function data based on strategy.
 * @tparam Initializer Transfer function initialization strategy
 * @return Vector of transfer function arrays
 */
template <typename Initializer>
std::vector<
    std::array<std::array<type_real, n_quad_intersection>, n_quad_element> >
init_transfer_function();

/**
 * @brief Initialize zero transfer function.
 * @return Vector containing zero-initialized transfer function
 */
template <>
std::vector<
    std::array<std::array<type_real, n_quad_intersection>, n_quad_element> >
init_transfer_function<TransferFunctionInitializer2D::Zero>() {
  return std::vector<
      std::array<std::array<type_real, n_quad_intersection>, n_quad_element> >(
      1, [] {
        std::array<std::array<type_real, n_quad_intersection>, n_quad_element>
            arr{};
        for (auto &inner : arr) {
          inner.fill(static_cast<type_real>(0.0));
        }
        return arr;
      }());
}

/**
 * @brief Test transfer function container.
 * @tparam Initializer Transfer function initialization strategy
 */
template <typename Initializer> struct TransferFunction2D {
  using TransferFunctionInitializer = Initializer;

private:
  std::vector<
      std::array<std::array<type_real, n_quad_intersection>, n_quad_element> >
      _transfer_function;
  using TransferFunctionView =
      Kokkos::View<type_real *[n_quad_element][n_quad_intersection],
                   memory_space>;

public:
  /**
   * @brief Construct transfer function with initializer.
   * @param initializer Initialization strategy
   */
  TransferFunction2D(const Initializer &initializer)
      : _transfer_function(init_transfer_function<Initializer>()) {}

  /**
   * @brief Get Kokkos view of transfer function data.
   * @return Kokkos view for device access
   */
  TransferFunctionView get_view() const {
    TransferFunctionView view("transfer_function_view",
                              _transfer_function.size());
    auto host_view = Kokkos::create_mirror_view(view);
    for (size_t i = 0; i < _transfer_function.size(); ++i) {
      for (size_t j = 0; j < n_quad_element; ++j) {
        for (size_t k = 0; k < n_quad_intersection; ++k) {
          host_view(i, j, k) = _transfer_function[i][j][k];
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  int get_edges() const { return _transfer_function.size(); }

  /**
   * @brief Access transfer function values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Intersection quadrature index
   * @return Reference to transfer function value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _transfer_function[i][j][k];
  }
};

/**
 * @brief Initialize field data based on strategy.
 * @tparam Initializer Field initialization strategy
 * @return Vector of field arrays
 */
template <typename Initializer>
std::vector<std::array<std::array<type_real, n_components>, n_quad_element> >
init_function();

/**
 * @brief Initialize uniform field.
 * @return Vector containing uniform field values
 */
template <>
std::vector<std::array<std::array<type_real, n_components>, n_quad_element> >
init_function<EdgeFunctionInitializer2D::Uniform>() {
  return std::vector<
      std::array<std::array<type_real, n_components>, n_quad_element> >(1, [] {
    std::array<std::array<type_real, n_components>, n_quad_element> arr{};
    for (auto &inner : arr) {
      inner.fill(static_cast<type_real>(1.0));
    }
    return arr;
  }());
}

/**
 * @brief Test field container.
 * @tparam Initializer Field initialization strategy
 */
template <typename Initializer> struct EdgeFunction2D {
  using FunctionInitializer = Initializer;

private:
  std::vector<std::array<std::array<type_real, n_components>, n_quad_element> >
      _field;
  using FieldView =
      Kokkos::View<type_real *[n_quad_element][n_components], memory_space>;

public:
  /**
   * @brief Construct field with initializer.
   * @param initializer Initialization strategy
   */
  EdgeFunction2D(const FunctionInitializer &initializer)
      : _field(init_function<FunctionInitializer>()) {}

  /**
   * @brief Get Kokkos view of field data.
   * @return Kokkos view for device access
   */
  FieldView get_view() const {
    FieldView view("field_view", _field.size());
    auto host_view = Kokkos::create_mirror_view(view);
    for (size_t i = 0; i < _field.size(); ++i) {
      for (size_t j = 0; j < n_quad_element; ++j) {
        for (size_t k = 0; k < n_components; ++k) {
          host_view(i, j, k) = _field[i][j][k];
        }
      }
    }
    Kokkos::deep_copy(view, host_view);
    return view;
  }

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  int get_edges() const { return _field.size(); }

  /**
   * @brief Access field values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Component index
   * @return Reference to field value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _field[i][j][k];
  }
};

/**
 * @brief Compute expected result of transfer function operation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param field Input field data
 * @return Expected transferred field values
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
std::vector<std::array<type_real, n_components> >
compute_transferred_function(const TransferFunction2D &transfer_function,
                             const EdgeFunction2D &field) {
  const int n_edges = transfer_function.get_edges();
  std::vector<std::array<type_real, n_components> > result_field(
      n_edges, std::array<type_real, n_components>{ 0.0 });
  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < n_quad_intersection; ++j) {
      for (int k = 0; k < n_components; ++k) {
        for (int l = 0; l < n_quad_element; ++l) {
          result_field[i][k] += transfer_function(i, j, l) * field(l, j, k);
        }
      }
    }
  }
  return result_field;
}

/**
 * @brief Compute transferred field for zero transfer function and uniform
 * field.
 * @param transfer_function Zero transfer function
 * @param field Uniform field
 * @return Zero field result
 */
template <>
std::vector<std::array<type_real, n_components> > compute_transferred_function(
    const TransferFunction2D<TransferFunctionInitializer2D::Zero>
        &transfer_function,
    const EdgeFunction2D<EdgeFunctionInitializer2D::Uniform> &field) {
  // Result field is a zero field
  const int n_edges = field.get_edges();
  return std::vector<std::array<type_real, n_components> >(n_edges, [] {
    std::array<type_real, n_components> arr{};
    arr.fill(static_cast<type_real>(0.0));
    return arr;
  }());
}

/**
 * @brief Execute transfer function test with validation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param function Input function data
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
void execute(const TransferFunction2D &transfer_function,
             const EdgeFunction2D &function) {
  auto expected = compute_transferred_function(transfer_function, function);

  const int n_edges = transfer_function.get_edges();
  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, n_quad_intersection, n_quad_element,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, memory_space, Kokkos::MemoryTraits<> >;
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, n_quad_element, n_components, false,
      memory_space, Kokkos::MemoryTraits<> >;

  const auto transfer_function_view = transfer_function.get_view();
  const auto function_view = function.get_view();

  Kokkos::View<type_real *[n_components], memory_space> result_view(
      "result_view", n_edges);

  Kokkos::parallel_for(
      "transfer_function_test", Kokkos::TeamPolicy<>(n_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge = team_member.league_rank();
        const TransferFunctionType TF(Kokkos::subview(
            transfer_function_view, Kokkos::make_pair(iedge, iedge + 1),
            Kokkos::ALL(), Kokkos::ALL()));
        const FunctionType F(
            Kokkos::subview(function_view, Kokkos::make_pair(iedge, iedge + 1),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::transfer(
            ChunkEdgeIndex(1, team_member), TF, F,
            [&](const auto &index, const auto &point) {
              for (int icomp = 0; icomp < n_components; ++icomp) {
                Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                  result_view(index(0), icomp) = point(icomp);
                });
              }
            });
      });

  Kokkos::fence();

  auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);

  for (int i = 0; i < n_edges; ++i) {
    if (!specfem::utilities::is_close(result_host(i, 0), expected[i][0])) {
      ADD_FAILURE() << "Transfer function test failed at edge " << i
                    << ": expected " << expected[i][0] << "\n got "
                    << result_host(i, 0);
    }
  }
}

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct TransferFunctionTest2D : public ::testing::Test {
  using TransferFunctionInitializer = std::tuple_element_t<0, TestingTypes>;
  using FunctionInitializer = std::tuple_element_t<1, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        function(FunctionInitializer()) {}

  TransferFunction2D<TransferFunctionInitializer> transfer_function; /**< Test
                                                                        transfer
                                                                        function
                                                                      */
  EdgeFunction2D<FunctionInitializer> function; /**< Test field */
};

/** Test type combinations for parameterized testing */
using TransferFunctionTestTypes2D =
    ::testing::Types<std::tuple<TransferFunctionInitializer2D::Zero,
                                EdgeFunctionInitializer2D::Uniform> >;

TYPED_TEST_SUITE(TransferFunctionTest2D, TransferFunctionTestTypes2D);

TYPED_TEST(TransferFunctionTest2D, ExecuteTransferFunction) {
  execute(this->transfer_function, this->function);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
