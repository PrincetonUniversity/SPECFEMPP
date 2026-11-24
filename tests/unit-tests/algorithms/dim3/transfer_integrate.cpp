#include "specfem/algorithms/transfer_interpolate.hpp"

#include "specfem/datatype/accessor_type.hpp"
#include "specfem/datatype/chunk_face_view.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/element_coupling/attributes.hpp"
#include "specfem/point/acceleration.hpp"
#include "specfem/point/face_index.hpp"
#include "specfem/point/nonconforming_interface3d.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace transfer_integrate_test_impl {

constexpr int tid_ngll = 5;
constexpr int tid_ncomp = 3; // elastic guest/host values
constexpr int tid_chunk = 4;
constexpr auto tid_dim = specfem::element::dimension_tag::dim3;

using InterpolantsView =
    Kokkos::View<type_real *****, Kokkos::LayoutRight, Kokkos::HostSpace>;
using FluxView =
    Kokkos::View<type_real[tid_chunk][tid_ngll][tid_ngll][tid_ncomp],
                 Kokkos::LayoutRight, Kokkos::HostSpace>;
using ResultType = specfem::point::acceleration<specfem::tags::Tags<
    tid_dim, specfem::element::medium_tag::elastic, false /*UseSIMD*/>>;

inline specfem::point::face_index<tid_dim>
make_index(const int iface, const int ki, const int kj) {
  return specfem::point::face_index<tid_dim>(
      0, iface, ki, kj, 0, 0, 0, specfem::mesh_entity::dim3::type::back);
}

inline InterpolantsView make_interpolants(const int tid_npair) {
  return InterpolantsView("interpolants", tid_npair, tid_ngll, tid_ngll,
                          tid_ngll, 2);
}

/**
 * @brief Runs transfer_integrate at every guest node of pair 0 (chunk
 * position 0) and returns the results as [ki][kj][icomp].
 */
inline std::array<std::array<std::array<type_real, tid_ncomp>, tid_ngll>,
                  tid_ngll>
integrate_all(const InterpolantsView &interpolants, const FluxView &flux) {
  std::array<std::array<std::array<type_real, tid_ncomp>, tid_ngll>, tid_ngll>
      out{};
  for (int ki = 0; ki < tid_ngll; ++ki) {
    for (int kj = 0; kj < tid_ngll; ++kj) {
      ResultType result;
      const auto index = make_index(0, ki, kj);
      specfem::algorithms::transfer_integrate(index, index, interpolants, flux,
                                              result);
      for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
        out[ki][kj][icomp] = result(icomp);
      }
    }
  }
  return out;
}

} // namespace transfer_integrate_test_impl

/**
 * Interpolants of one host node are unit rows selecting guest node (1, 3); all
 * other host nodes have zero rows but nonzero flux. Only guest node (1, 3) may
 * receive the chosen flux. Different indices on the two axes catch swapped
 * axes; different values per component catch component mix-ups.
 */
TEST(TransferIntegrateTests, HandValue) {
  using namespace transfer_integrate_test_impl;
  constexpr int ci_sel = 2;
  constexpr int cj_sel = 4;
  constexpr int ki_sel = 1;
  constexpr int kj_sel = 3;
  const type_real chosen[tid_ncomp] = { 2.5, -4.0, 7.0 };

  auto interpolants = make_interpolants(1);
  Kokkos::deep_copy(interpolants, 0); // zero rows at all other host nodes
  for (int k = 0; k < tid_ngll; ++k) {
    interpolants(0, ci_sel, cj_sel, k, 0) = (k == ki_sel) ? 1 : 0;
    interpolants(0, ci_sel, cj_sel, k, 1) = (k == kj_sel) ? 1 : 0;
  }

  FluxView flux("flux");
  for (int ci = 0; ci < tid_ngll; ++ci) {
    for (int cj = 0; cj < tid_ngll; ++cj) {
      for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
        flux(0, ci, cj, icomp) =
            (ci == ci_sel && cj == cj_sel)
                ? chosen[icomp]
                : static_cast<type_real>(100 + 10 * ci + cj + 0.5 * icomp);
      }
    }
  }

  const auto out = integrate_all(interpolants, flux);

  for (int ki = 0; ki < tid_ngll; ++ki) {
    for (int kj = 0; kj < tid_ngll; ++kj) {
      for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
        if (ki == ki_sel && kj == kj_sel) {
          // 1 * 1 * chosen is exact: the other nodes add 0 * junk == 0.
          EXPECT_NEAR(out[ki][kj][icomp], chosen[icomp], 1e-6)
              << "guest node (" << ki << ", " << kj << "), component " << icomp;
        } else {
          // Every product contains an exact 0 interpolant factor (or only
          // 0 * junk), so the sum is exactly zero.
          EXPECT_EQ(out[ki][kj][icomp], type_real(0))
              << "guest node (" << ki << ", " << kj << "), component " << icomp;
        }
      }
    }
  }
}

TEST(TransferIntegrateTests, ZeroRowsContributeNothing) {
  using namespace transfer_integrate_test_impl;
  auto interpolants = make_interpolants(1);
  Kokkos::deep_copy(interpolants, 0);

  FluxView flux("flux");
  for (int ci = 0; ci < tid_ngll; ++ci) {
    for (int cj = 0; cj < tid_ngll; ++cj) {
      for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
        flux(0, ci, cj, icomp) =
            static_cast<type_real>(1 + ci - 2 * cj + 3 * icomp);
      }
    }
  }

  const auto out = integrate_all(interpolants, flux);
  for (int ki = 0; ki < tid_ngll; ++ki) {
    for (int kj = 0; kj < tid_ngll; ++kj) {
      for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
        // Exact: finite flux times exact zero interpolants is exactly zero.
        EXPECT_EQ(out[ki][kj][icomp], type_real(0))
            << "guest node (" << ki << ", " << kj << "), component " << icomp;
      }
    }
  }
}

/**
 * Adjointness: with T the matrix of transfer_interpolate,
 * sum_c q(c) (T u)(c) == sum_k u(k) (T^T q)(k) summed over pairs and
 * components. 10 pairs with chunk size 4 give chunks {0-3}, {4-7}, {8-9}.
 *
 * Data: std::mt19937 seeded with 20240607, uniform in [0, 1) (all positive so
 * the sums do not cancel and the magnitude check is meaningful).
 *
 * Tolerance: type_real may be single precision. Each node value is a sum of
 * 25 products and the adjoint of a sum of 25 more, each with relative
 * rounding error <= ~5 epsilon, so 100 * epsilon * (sum of absolute terms)
 * bounds the accumulated error of the double-precision totals with a wide
 * margin, while staying far below the signal (all terms positive).
 */
TEST(TransferIntegrateTests, IsTransposeOfTransferInterpolate) {
  using namespace transfer_integrate_test_impl;
  constexpr int tid_npair = 10;
  constexpr int tid_nchunk = (tid_npair + tid_chunk - 1) / tid_chunk;
  constexpr unsigned seed = 20240607;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  auto interpolants = make_interpolants(tid_npair);
  Kokkos::View<type_real *[tid_ngll][tid_ngll][tid_ncomp], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      u("u", tid_npair);
  Kokkos::View<type_real *[tid_ngll][tid_ngll][tid_ncomp], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      q("q", tid_npair);
  for (int p = 0; p < tid_npair; ++p) {
    for (int a = 0; a < tid_ngll; ++a) {
      for (int b = 0; b < tid_ngll; ++b) {
        for (int g = 0; g < tid_ngll; ++g) {
          for (int axis = 0; axis < 2; ++axis) {
            interpolants(p, a, b, g, axis) = static_cast<type_real>(dist(rng));
          }
        }
        for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
          u(p, a, b, icomp) = static_cast<type_real>(dist(rng));
          q(p, a, b, icomp) = static_cast<type_real>(dist(rng));
        }
      }
    }
  }

  using InterpolantsAccessor = typename specfem::point::nonconforming_interface<
      tid_dim, tid_ngll,
      specfem::element_coupling::interface_tag::acoustic_elastic,
      specfem::element::boundary_tag::none,
      specfem::element_coupling::flux_scheme_tag::natural>;
  using InterpolantsRows =
      typename InterpolantsAccessor::template tensor_type<type_real, tid_ngll,
                                                          2>;
  using ChunkField = specfem::datatype::VectorChunkFaceViewType<
      type_real, tid_dim, tid_chunk, tid_ngll, tid_ncomp, false /*UseSIMD*/,
      Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<>,
      Kokkos::LayoutRight>;
  using ChunkStorage =
      Kokkos::View<type_real[tid_chunk][tid_ngll][tid_ngll][tid_ncomp],
                   Kokkos::LayoutRight, Kokkos::HostSpace>;

  double lhs = 0; // sum q(c) (T u)(c)
  double rhs = 0; // sum u(k) (T^T q)(k)
  double abs_sum = 0;

  for (int ichunk = 0; ichunk < tid_nchunk; ++ichunk) {
    const int chunk_start = ichunk * tid_chunk;
    const int this_chunk_size = std::min(tid_chunk, tid_npair - chunk_start);

    // chunk copies of u (field to interpolate) and q (flux to integrate)
    ChunkStorage u_chunk("u_chunk");
    FluxView q_chunk("q_chunk");
    Kokkos::deep_copy(u_chunk, 0);
    Kokkos::deep_copy(q_chunk, 0);
    for (int il = 0; il < this_chunk_size; ++il) {
      for (int a = 0; a < tid_ngll; ++a) {
        for (int b = 0; b < tid_ngll; ++b) {
          for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
            u_chunk(il, a, b, icomp) = u(chunk_start + il, a, b, icomp);
            q_chunk(il, a, b, icomp) = q(chunk_start + il, a, b, icomp);
          }
        }
      }
    }
    const ChunkField F(u_chunk);

    for (int il = 0; il < this_chunk_size; ++il) {
      const int ipair = chunk_start + il;

      // (T u)(c) via the production transfer_interpolate
      for (int ci = 0; ci < tid_ngll; ++ci) {
        for (int cj = 0; cj < tid_ngll; ++cj) {
          InterpolantsRows rows;
          for (int g = 0; g < tid_ngll; ++g) {
            rows(g, 0) = interpolants(ipair, ci, cj, g, 0);
            rows(g, 1) = interpolants(ipair, ci, cj, g, 1);
          }
          const InterpolantsAccessor NCI({ 1.0 }, { 0, 0, 0 }, rows);
          specfem::datatype::VectorPointViewType<type_real, tid_ncomp, false>
              interpolated;
          specfem::algorithms::transfer_interpolate(make_index(il, ci, cj), NCI,
                                                    F, interpolated);
          for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
            const double term = static_cast<double>(q(ipair, ci, cj, icomp)) *
                                static_cast<double>(interpolated(icomp));
            lhs += term;
            abs_sum += std::abs(term);
          }
        }
      }

      // (T^T q)(k) via transfer_integrate
      for (int ki = 0; ki < tid_ngll; ++ki) {
        for (int kj = 0; kj < tid_ngll; ++kj) {
          ResultType integrated;
          specfem::algorithms::transfer_integrate(
              make_index(ipair, ki, kj), make_index(il, ki, kj), interpolants,
              q_chunk, integrated);
          for (int icomp = 0; icomp < tid_ncomp; ++icomp) {
            rhs += static_cast<double>(u(ipair, ki, kj, icomp)) *
                   static_cast<double>(integrated(icomp));
          }
        }
      }
    }
  }

  const double tol =
      100.0 * std::numeric_limits<type_real>::epsilon() * abs_sum;
  EXPECT_GT(std::abs(lhs), 1000.0 * tol)
      << "signal must be far above the tolerance (lhs = " << lhs << ")";
  EXPECT_NEAR(lhs, rhs, tol)
      << "relative difference " << std::abs(lhs - rhs) / std::abs(lhs);
  RecordProperty("relative_difference", std::abs(lhs - rhs) / std::abs(lhs));
}
