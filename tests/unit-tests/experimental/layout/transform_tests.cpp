#ifdef SPECFEM_LAYOUT_FORCE_DEBUG
#undef NDEBUG
#define TRANSFORM_SUITE TransformDebug
#else
#define TRANSFORM_SUITE Transform
#endif
#include "restricted_mapping.hpp"
#include "specfem/experimental/layout/kokkos.hpp"
#include "specfem/experimental/layout/transform.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <type_traits>

namespace hl = specfem::experimental::layout;

TEST(TRANSFORM_SUITE, CoordinateOracle) {
  const auto parent = hl::make_layout(
      hl::make_tuple(hl::make_tuple(hl::c<2>, hl::make_tuple(3, hl::c<2>)), 2),
      hl::make_tuple(hl::make_tuple(hl::c<1>, hl::make_tuple(4, hl::c<16>)),
                     40));
  const auto transformed = hl::permute<hl::ModePath<0, 1>, 1, 0>(parent);
  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(hl::get<0>(
                         hl::get<1>(hl::get<0>(transformed.shape()))))>,
                     std::integral_constant<int, 2>>);
  std::set<std::size_t> offsets;
  for (int d = 0; d < 2; ++d) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int a = 0; a < 2; ++a) {
          const auto nested =
              hl::make_tuple(hl::make_tuple(a, hl::make_tuple(c, b)), d);
          const auto scalar = hl::make_tuple(a + 2 * (c + 2 * b), d);
          const std::size_t expected = a + 4 * b + 16 * c + 40 * d;
          EXPECT_TRUE(transformed.valid(nested));
          EXPECT_EQ(transformed.offset(nested), expected);
          EXPECT_EQ(transformed.offset(scalar), expected);
          EXPECT_EQ(parent.offset(hl::make_tuple(
                        hl::make_tuple(a, hl::make_tuple(b, c)), d)),
                    expected);
          offsets.insert(transformed.offset(scalar));
        }
      }
    }
  }
  EXPECT_EQ(offsets.size(), 24);
  EXPECT_EQ(transformed.required_span_size(), parent.required_span_size());
  EXPECT_FALSE(transformed.valid(hl::make_tuple(12, 0)));
  EXPECT_FALSE(transformed.valid(hl::make_tuple(-1, 0)));
}

TEST(TRANSFORM_SUITE, CompositionsAndOffsets) {
  constexpr auto static_parent = hl::make_layout(
      hl::make_tuple(hl::c<2>, hl::c<3>), hl::make_tuple(hl::c<1>, hl::c<8>));
  constexpr auto static_group = hl::group<hl::ModePath<>, 0, 2>(static_parent);
  static_assert(static_group.offset(hl::make_tuple(hl::c<5>)) == 17);
  constexpr auto static_permuted =
      hl::permute<hl::ModePath<0>, 1, 0>(static_group);
  constexpr auto static_flat = hl::ungroup<hl::ModePath<0>>(static_permuted);
  static_assert(static_flat.offset(hl::make_tuple(hl::c<2>, hl::c<1>)) == 17);

  const auto parent = hl::make_layout(hl::make_tuple(hl::c<2>, 3, hl::c<2>),
                                      hl::make_tuple(hl::c<1>, 8, hl::c<32>));
  const auto p = hl::permute<hl::ModePath<>, 2, 0, 1>(parent);
  const auto g = hl::group<hl::ModePath<>, 0, 2>(p);
  const auto nested = hl::group<hl::ModePath<0>, 0, 1>(g);
  const auto ungrouped = hl::ungroup<hl::ModePath<0, 0>>(nested);
  const auto restored = hl::ungroup<hl::ModePath<0>>(ungrouped);
  static_assert(decltype(g)::rank() == 2);
  static_assert(decltype(g.template extent<0>())::value == 4);
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 2; ++c) {
        const std::size_t expected = a + 8 * b + 32 * c;
        EXPECT_EQ(g.offset(hl::make_tuple(c + 2 * a, b)), expected);
        EXPECT_EQ(nested.offset(hl::make_tuple(c + 2 * a, b)), expected);
        EXPECT_EQ(restored.offset(hl::make_tuple(c, a, b)), expected);
        EXPECT_EQ(parent.offset(hl::make_tuple(a, b, c)), expected);
      }
    }
  }
  EXPECT_EQ(g.required_span_size(), 50);
  EXPECT_EQ(g.envelope_size(), 12);
}

TEST(TRANSFORM_SUITE, EmptySingletonAndScalar) {
  const auto scalar = hl::make_layout(hl::make_tuple(), hl::make_tuple());
  const auto identity = hl::permute<hl::ModePath<>>(scalar);
  EXPECT_TRUE(identity.valid(hl::make_tuple()));
  EXPECT_EQ(identity.offset(hl::make_tuple()), 0);
  const auto unit = hl::make_layout(hl::make_tuple(hl::make_tuple()),
                                    hl::make_tuple(hl::make_tuple()));
  const auto removed = hl::ungroup<hl::ModePath<0>>(unit);
  EXPECT_TRUE(removed.valid(hl::make_tuple()));
  EXPECT_EQ(removed.required_span_size(), 1);
  EXPECT_EQ(removed.offset(hl::make_tuple()), 0);
  for (int zero_axis = 0; zero_axis < 3; ++zero_axis) {
    const auto parent =
        hl::make_layout(hl::make_tuple(zero_axis == 0 ? 0 : 2,
                                       hl::make_tuple(zero_axis == 1 ? 0 : 3,
                                                      zero_axis == 2 ? 0 : 1)),
                        hl::make_tuple(1, hl::make_tuple(4, 16)));
    const auto p = hl::permute<hl::ModePath<>, 1, 0>(parent);
    const auto g = hl::group<hl::ModePath<>, 0, 2>(p);
    const auto u = hl::ungroup<hl::ModePath<0>>(g);
    EXPECT_TRUE(u.empty());
    EXPECT_EQ(u.required_span_size(), 0);
    EXPECT_FALSE(g.valid(hl::make_tuple(0)));
    const auto view = hl::make_view(static_cast<int *>(nullptr), 0, parent);
    EXPECT_TRUE((hl::group<hl::ModePath<>, 0, 2>(view).layout().empty()));
  }
  const auto disabled =
      hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(1, 2), false);
  const auto disabled_group = hl::group<hl::ModePath<>, 0, 2>(disabled);
  const auto disabled_permute = hl::permute<hl::ModePath<>, 1, 0>(disabled);
  const auto disabled_ungroup = hl::ungroup<hl::ModePath<0>>(disabled_group);
  EXPECT_FALSE(disabled_group.valid(hl::make_tuple(0)));
  EXPECT_TRUE(disabled_permute.empty());
  EXPECT_EQ(disabled_ungroup.required_span_size(), 0);
  const auto singleton =
      hl::make_layout(hl::make_tuple(1, 3), hl::make_tuple(0, 2));
  const auto g = hl::group<hl::ModePath<>, 0, 2>(singleton);
  EXPECT_EQ(g.offset(hl::make_tuple(2)), 4);
  EXPECT_FALSE(g.valid(hl::make_tuple(3)));
}

TEST(TRANSFORM_SUITE, PreservesInheritedValidity) {
  const specfem::testing::layout_transform_impl::RestrictedMapping parent;
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(parent);
  const auto g = hl::group<hl::ModePath<>, 0, 2>(parent);
  const auto u = hl::ungroup<hl::ModePath<0>>(parent);
  const auto nested = hl::permute<hl::ModePath<0>, 1, 0>(g);
  const auto flat =
      hl::ungroup<hl::ModePath<1>>(hl::ungroup<hl::ModePath<0>>(nested));
  int data[12];
  for (auto &value : data) {
    value = -1;
  }
  const auto view =
      hl::permute<hl::ModePath<>, 1, 0>(hl::make_view(data, 12, parent));
  for (int c = 0; c < 2; ++c) {
    for (int b = 0; b < 3; ++b) {
      for (int a = 0; a < 2; ++a) {
        // Exclude (1,0,0), (0,2,1), and (1,2,1) in original axes.
        const bool expected =
            !(c == 0 && b == 0 && a == 1) && !(c == 1 && b == 2);
        const int offset = a + 2 * b + 6 * c;
        const auto original = hl::make_tuple(hl::make_tuple(a, b), c);
        EXPECT_EQ(parent.valid(original), expected);
        EXPECT_EQ(p.valid(hl::make_tuple(c, a + 2 * b)), expected);
        EXPECT_EQ(p.valid(hl::make_tuple(c, hl::make_tuple(a, b))), expected);
        EXPECT_EQ(g.valid(hl::make_tuple(offset)), expected);
        EXPECT_EQ(g.valid(hl::make_tuple(original)), expected);
        EXPECT_EQ(u.valid(hl::make_tuple(a, b, c)), expected);
        EXPECT_EQ(nested.valid(hl::make_tuple(c + 2 * (a + 2 * b))), expected);
        EXPECT_EQ(nested.valid(
                      hl::make_tuple(hl::make_tuple(c, hl::make_tuple(a, b)))),
                  expected);
        EXPECT_EQ(flat.valid(hl::make_tuple(c, a, b)), expected);
        const auto coordinate = hl::make_tuple(c, a + 2 * b);
        const bool valid = view.layout().valid(coordinate);
        EXPECT_EQ(valid, expected);
        if (valid) {
          view(coordinate) = offset;
        }
        EXPECT_EQ(data[offset], expected ? offset : -1);
        if (expected) {
          EXPECT_EQ(flat.offset(hl::make_tuple(c, a, b)), offset);
        }
      }
    }
  }
  EXPECT_FALSE(p.valid(hl::make_tuple(2, 0)));
  EXPECT_FALSE(g.valid(hl::make_tuple(-1)));
  EXPECT_FALSE(nested.valid(hl::make_tuple(12)));
}

TEST(TRANSFORM_SUITE, AliasingAndLifetime) {
  const auto retained = [] {
    Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace> native(
        "retained", 2, 3);
    const auto original = hl::make_view(native);
    const auto p = hl::permute<hl::ModePath<>, 1, 0>(original);
    const auto g = hl::group<hl::ModePath<>, 0, 2>(p);
    const auto u = hl::ungroup<hl::ModePath<0>>(g);
    u(hl::make_tuple(2, 1)) = 51;
    EXPECT_EQ(&native(1, 2), &u(hl::make_tuple(2, 1)));
    EXPECT_EQ(native(1, 2), 51);
    return u;
  }();
  Kokkos::View<int *, Kokkos::HostSpace> other("other", 6);
  Kokkos::deep_copy(other, -1);
  EXPECT_EQ(retained(hl::make_tuple(2, 1)), 51);
  retained(hl::make_tuple(2, 1)) = 73;
  EXPECT_EQ(retained(hl::make_tuple(2, 1)), 73);

  int data[6]{};
  const auto mapping =
      hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(1, 2));
  const auto raw = hl::make_view(data, 6, mapping);
  const auto p = hl::permute<hl::ModePath<>, 1, 0>(raw);
  p(hl::make_tuple(2, 1)) = 19;
  EXPECT_EQ(&p(hl::make_tuple(2, 1)), data + 5);
  using unmanaged = Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const unmanaged native(data, 2, 3);
  const auto borrowed = hl::group<hl::ModePath<>, 0, 2>(hl::make_view(native));
  borrowed(hl::make_tuple(5)) = 29;
  EXPECT_EQ(native(1, 2), 29);
  const auto read_only = hl::group<hl::ModePath<>, 0, 2>(
      hl::make_view(static_cast<const int *>(data), 6, mapping));
  static_assert(
      std::is_same_v<decltype(read_only(hl::make_tuple(5))), const int &>);
  EXPECT_EQ(read_only(hl::make_tuple(5)), 29);
}

namespace specfem::testing::layout_transform_impl {
void expect_kernel_reorganization();
} // namespace specfem::testing::layout_transform_impl

void specfem::testing::layout_transform_impl::expect_kernel_reorganization() {
  Kokkos::View<int ***, Kokkos::LayoutRight> native("transform_kernel", 2, 3,
                                                    2);
  Kokkos::deep_copy(native, -1);
  const auto captured = hl::group<hl::ModePath<>, 0, 2>(hl::make_view(native));
  Kokkos::parallel_for(
      "transform_kernel", 12, KOKKOS_LAMBDA(int i) {
        const auto p = hl::permute<hl::ModePath<>, 1, 0>(captured);
        const auto u = hl::ungroup<hl::ModePath<1>>(p);
        const auto g = hl::group<hl::ModePath<>, 1, 2>(u);
        const auto q = hl::make_tuple(i % 2, i / 2);
        if (g.layout().valid(q)) {
          g(q) = i + 10;
        }
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, native);
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 2; ++c) {
        EXPECT_EQ(host(a, b, c), c + 2 * (a + 2 * b) + 10);
      }
    }
  }
}

TEST(TRANSFORM_SUITE, KernelReorganization) {
  specfem::testing::layout_transform_impl::expect_kernel_reorganization();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  if (argc == 2 && std::strcmp(argv[1], "--invalid-device-coordinate") == 0) {
    {
      Kokkos::View<std::size_t *> result("invalid_result", 1);
      Kokkos::parallel_for(
          "invalid_transform", 1, KOKKOS_LAMBDA(int i) {
            const auto parent =
                hl::make_layout(hl::make_tuple(2, 3), hl::make_tuple(1, 2));
            const auto g = hl::group<hl::ModePath<>, 0, 2>(parent);
            result(i) = g.offset(hl::make_tuple(i + 6));
          });
      Kokkos::fence();
    }
    Kokkos::finalize();
    return 0;
  }
  const int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
