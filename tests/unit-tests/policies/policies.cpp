#include "../SPECFEM_Environment.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "execution/range_iterator.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "parallel_configuration/range_config.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Base fixture for common functionality
class ChunkedDomainIteratorTestBase {
public:
  static constexpr int default_ngll = 5;
};

// Test parameter structs
struct RangePolicyTestParams {
  int nglob;
  std::string name;

  RangePolicyTestParams(int n, const char *test_name)
      : nglob(n), name(test_name) {}
};

struct ChunkElementPolicyTestParams {
  int nspec;
  int ngllz;
  int ngllx;
  std::string name;

  ChunkElementPolicyTestParams(int nspec, int ngllz, int ngllx,
                               const char *test_name)
      : nspec(nspec), ngllz(ngllz), ngllx(ngllx), name(test_name) {}
};

struct ChunkElementPolicy3DTestParams {
  int nspec;
  int ngllz;
  int nglly;
  int ngllx;
  std::string name;

  ChunkElementPolicy3DTestParams(int nspec, int ngllz, int nglly, int ngllx,
                                 const char *test_name)
      : nspec(nspec), ngllz(ngllz), nglly(nglly), ngllx(ngllx),
        name(test_name) {}
};

// Operators for test naming
std::ostream &operator<<(std::ostream &os,
                         const RangePolicyTestParams &params) {
  os << params.name;
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const ChunkElementPolicyTestParams &params) {
  os << params.name;
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const ChunkElementPolicy3DTestParams &params) {
  os << params.name;
  return os;
}

// ------------------------------------------------------------------------

template <typename ParallelConfig>
typename Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>::HostMirror
execute_range_policy(const int nglob) {
  specfem::execution::RangeIterator iterator(ParallelConfig(), nglob);
  using TestViewType = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  TestViewType test_view("test_view", nglob);
  TestViewType::HostMirror test_view_host =
      Kokkos::create_mirror_view(test_view);

  constexpr bool using_simd = ParallelConfig::simd::using_simd;

  using PointIndex = specfem::point::assembly_index<using_simd>;

  // initialize test_view
  Kokkos::parallel_for(
      "initialize_test_view",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nglob),
      KOKKOS_LAMBDA(const int iglob) { test_view(iglob) = 0; });

  Kokkos::fence();

  specfem::execution::for_all(
      "execute_range_policy", iterator,
      KOKKOS_LAMBDA(
          const typename decltype(iterator)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        const auto l_test_view = test_view;
        constexpr bool is_simd = using_simd;
        if constexpr (is_simd) {
          using tag_type = typename ParallelConfig::simd::tag_type;
          using datatype = typename ParallelConfig::simd::datatype;
          using mask_type = typename ParallelConfig::simd::mask_type;
          mask_type mask([&](std::size_t lane) { return index.mask(lane); });
          datatype data(0);
          Kokkos::Experimental::where(mask, data)
              .copy_from(&l_test_view(index.iglob), tag_type());

          data = data + datatype(1);
          Kokkos::Experimental::where(mask, data)
              .copy_to(&l_test_view(index.iglob), tag_type());
        } else {
          l_test_view(index.iglob) += 1;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(test_view_host, test_view);
  return test_view_host;
}

template <typename ParallelConfig>
typename Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>::HostMirror
execute_chunk_element_policy(const int nspec, const int ngllz,
                             const int ngllx) {

  constexpr bool using_simd = ParallelConfig::simd::using_simd;

  constexpr auto dimension = specfem::dimension::type::dim2;

  const specfem::mesh_entity::element_grid<dimension> element_grid(ngllz,
                                                                   ngllx);

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> elements("elements",
                                                              nspec);

  // Initialize elements
  Kokkos::parallel_for(
      "initialize_elements",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nspec),
      KOKKOS_LAMBDA(const int ispec) { elements(ispec) = ispec; });

  Kokkos::fence();

  specfem::execution::ChunkedDomainIterator policy(ParallelConfig(), elements,
                                                   element_grid);

  using TestViewType = Kokkos::View<type_real ***, Kokkos::LayoutLeft,
                                    Kokkos::DefaultExecutionSpace>;

  TestViewType test_view("test_view", nspec, ngllz, ngllx);
  TestViewType::HostMirror test_view_host =
      Kokkos::create_mirror_view(test_view);

  // initialize test_view
  Kokkos::parallel_for(
      "initialize_test_view",
      Kokkos::MDRangePolicy<Kokkos::Rank<3> >({ 0, 0, 0 },
                                              { nspec, ngllz, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int ix) {
        test_view(ispec, iz, ix) = 0;
      });

  Kokkos::fence();

  using PointIndex = specfem::point::index<dimension, using_simd>;

  specfem::execution::for_all(
      "specfem::tests::execution::chunked_domain", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int ix = point_index.ix;
        constexpr bool is_simd = using_simd;
        const auto l_test_view = test_view;

        if constexpr (is_simd) {
          using mask_type = typename ParallelConfig::simd::mask_type;
          mask_type mask(
              [&](std::size_t lane) { return point_index.mask(lane); });
          using tag_type = typename ParallelConfig::simd::tag_type;
          using datatype = typename ParallelConfig::simd::datatype;
          datatype data(0);
          Kokkos::Experimental::where(mask, data)
              .copy_from(&l_test_view(ispec, iz, ix), tag_type());

          data = data + datatype(1);
          Kokkos::Experimental::where(mask, data)
              .copy_to(&l_test_view(ispec, iz, ix), tag_type());
        } else {
          l_test_view(ispec, iz, ix) += 1;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(test_view_host, test_view);
  return test_view_host;
}

template <typename ParallelConfig>
typename Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                      Kokkos::DefaultExecutionSpace>::HostMirror
execute_chunk_element_policy_3d(const int nspec, const int ngllz,
                                const int nglly, const int ngllx) {

  constexpr bool using_simd = ParallelConfig::simd::using_simd;

  constexpr auto dimension = specfem::dimension::type::dim3;

  const specfem::mesh_entity::element element_grid(ngllz, nglly, ngllx);

  Kokkos::View<int *, Kokkos::DefaultExecutionSpace> elements("elements",
                                                              nspec);

  // Initialize elements
  Kokkos::parallel_for(
      "initialize_elements",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nspec),
      KOKKOS_LAMBDA(const int ispec) { elements(ispec) = ispec; });

  Kokkos::fence();

  specfem::execution::ChunkedDomainIterator policy(ParallelConfig(), elements,
                                                   element_grid);

  using TestViewType = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                    Kokkos::DefaultExecutionSpace>;

  TestViewType test_view("test_view", nspec, ngllz, nglly, ngllx);
  TestViewType::HostMirror test_view_host =
      Kokkos::create_mirror_view(test_view);

  // initialize test_view
  Kokkos::parallel_for(
      "initialize_test_view",
      Kokkos::MDRangePolicy<Kokkos::Rank<4> >({ 0, 0, 0, 0 },
                                              { nspec, ngllz, nglly, ngllx }),
      KOKKOS_LAMBDA(const int ispec, const int iz, const int iy, const int ix) {
        test_view(ispec, iz, iy, ix) = 0;
      });

  Kokkos::fence();

  using PointIndex = specfem::point::index<dimension, using_simd>;

  specfem::execution::for_all(
      "specfem::tests::execution::chunked_domain_3d", policy,
      KOKKOS_LAMBDA(
          const typename decltype(policy)::base_index_type &iterator_index) {
        const auto point_index = iterator_index.get_index();
        const int ispec = point_index.ispec;
        const int iz = point_index.iz;
        const int iy = point_index.iy;
        const int ix = point_index.ix;
        constexpr bool is_simd = using_simd;
        const auto l_test_view = test_view;

        if constexpr (is_simd) {
          using mask_type = typename ParallelConfig::simd::mask_type;
          mask_type mask(
              [&](std::size_t lane) { return point_index.mask(lane); });
          using tag_type = typename ParallelConfig::simd::tag_type;
          using datatype = typename ParallelConfig::simd::datatype;
          datatype data(0);
          Kokkos::Experimental::where(mask, data)
              .copy_from(&l_test_view(ispec, iz, iy, ix), tag_type());

          data = data + datatype(1);
          Kokkos::Experimental::where(mask, data)
              .copy_to(&l_test_view(ispec, iz, iy, ix), tag_type());
        } else {
          l_test_view(ispec, iz, iy, ix) += 1;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(test_view_host, test_view);
  return test_view_host;
}

// Parameterized test classes
class RangePolicyTest : public ChunkedDomainIteratorTestBase,
                        public ::testing::TestWithParam<RangePolicyTestParams> {
protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common cleanup if needed
  }
};

class ChunkElementPolicyTest
    : public ChunkedDomainIteratorTestBase,
      public ::testing::TestWithParam<ChunkElementPolicyTestParams> {
protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common cleanup if needed
  }
};

class ChunkElementPolicy3DTest
    : public ChunkedDomainIteratorTestBase,
      public ::testing::TestWithParam<ChunkElementPolicy3DTestParams> {
protected:
  void SetUp() override {
    // Common setup if needed
  }

  void TearDown() override {
    // Common cleanup if needed
  }
};

TEST_P(RangePolicyTest, VisitAllPoints) {
  const auto params = GetParam();
  const int nglob = params.nglob;

  using ParallelConfig = specfem::parallel_configuration::default_range_config<
      specfem::datatype::simd<type_real, false>, Kokkos::DefaultExecutionSpace>;
  using SimdParallelConfig =
      specfem::parallel_configuration::default_range_config<
          specfem::datatype::simd<type_real, true>,
          Kokkos::DefaultExecutionSpace>;

  const auto check_test_view = [&](const auto &test_view,
                                   const std::string &error) {
    for (int iglob = 0; iglob < nglob; iglob++) {
      EXPECT_EQ(test_view(iglob), 1)
          << "Range policy failed at iglob=" << iglob
          << " for test: " << params.name << " - " << error;
    }
  };

  auto test_view = execute_range_policy<ParallelConfig>(nglob);
  auto simd_test_view = execute_range_policy<SimdParallelConfig>(nglob);

  check_test_view(test_view, "Error in RangePolicy with SIMD OFF");
  check_test_view(simd_test_view, "Error in RangePolicy with SIMD ON");
}

TEST_P(ChunkElementPolicyTest, VisitAllPoints) {
  const auto params = GetParam();
  const int nspec = params.nspec;
  const int ngllz = params.ngllz;
  const int ngllx = params.ngllx;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
      specfem::dimension::type::dim2, specfem::datatype::simd<type_real, false>,
      Kokkos::DefaultExecutionSpace>;

  using SimdParallelConfig =
      specfem::parallel_configuration::default_chunk_config<
          specfem::dimension::type::dim2,
          specfem::datatype::simd<type_real, true>,
          Kokkos::DefaultExecutionSpace>;

  const auto check_test_view = [&](const auto &test_view,
                                   const std::string &error) {
    for (int ispec = 0; ispec < nspec; ispec++) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int ix = 0; ix < ngllx; ix++) {
          EXPECT_EQ(test_view(ispec, iz, ix), 1)
              << "2D ChunkElement policy failed at (" << ispec << "," << iz
              << "," << ix << ")"
              << " for test: " << params.name << " - " << error;
        }
      }
    }
  };

  auto test_view =
      execute_chunk_element_policy<ParallelConfig>(nspec, ngllz, ngllx);
  auto simd_test_view =
      execute_chunk_element_policy<SimdParallelConfig>(nspec, ngllz, ngllx);

  check_test_view(test_view, "Error in ChunkElementPolicy with SIMD OFF");
  check_test_view(simd_test_view, "Error in ChunkElementPolicy with SIMD ON");
}

TEST_P(ChunkElementPolicy3DTest, VisitAllPoints) {
  const auto params = GetParam();
  const int nspec = params.nspec;
  const int ngllz = params.ngllz;
  const int nglly = params.nglly;
  const int ngllx = params.ngllx;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
      specfem::dimension::type::dim3, specfem::datatype::simd<type_real, false>,
      Kokkos::DefaultExecutionSpace>;

  using SimdParallelConfig =
      specfem::parallel_configuration::default_chunk_config<
          specfem::dimension::type::dim3,
          specfem::datatype::simd<type_real, true>,
          Kokkos::DefaultExecutionSpace>;

  const auto check_test_view = [&](const auto &test_view,
                                   const std::string &error) {
    for (int ispec = 0; ispec < nspec; ispec++) {
      for (int iz = 0; iz < ngllz; iz++) {
        for (int iy = 0; iy < nglly; iy++) {
          for (int ix = 0; ix < ngllx; ix++) {
            EXPECT_EQ(test_view(ispec, iz, iy, ix), 1)
                << "3D ChunkElement policy failed at (" << ispec << "," << iz
                << "," << iy << "," << ix << ")"
                << " for test: " << params.name << " - " << error;
          }
        }
      }
    }
  };

  auto test_view = execute_chunk_element_policy_3d<ParallelConfig>(
      nspec, ngllz, nglly, ngllx);
  auto simd_test_view = execute_chunk_element_policy_3d<SimdParallelConfig>(
      nspec, ngllz, nglly, ngllx);

  check_test_view(test_view, "Error in 3D ChunkElementPolicy with SIMD OFF");
  check_test_view(simd_test_view,
                  "Error in 3D ChunkElementPolicy with SIMD ON");
}

// Test instantiations with comprehensive parameter sets
INSTANTIATE_TEST_SUITE_P(
    Policies, RangePolicyTest,
    ::testing::Values(RangePolicyTestParams{ 100, "SmallRangeTest" },
                      RangePolicyTestParams{ 1000, "MediumRangeTest" },
                      RangePolicyTestParams{ 10000, "LargeRangeTest" },
                      RangePolicyTestParams{ 25000, "VeryLargeRangeTest" }));

INSTANTIATE_TEST_SUITE_P(
    Policies, ChunkElementPolicyTest,
    ::testing::Values(
        ChunkElementPolicyTestParams{ 10, 5, 5, "Small2DTest" },
        ChunkElementPolicyTestParams{ 100, 5, 5, "Medium2DTest" },
        ChunkElementPolicyTestParams{ 1000, 5, 5, "Large2DTest" },
        ChunkElementPolicyTestParams{ 10000, 5, 5, "VeryLarge2DTest" },
        ChunkElementPolicyTestParams{ 1024, 5, 5, "ExactChunkSize2DTest" },
        ChunkElementPolicyTestParams{ 50, 3, 3, "Small2DTest3x3" },
        ChunkElementPolicyTestParams{ 50, 7, 7, "Small2DTest7x7" }));

INSTANTIATE_TEST_SUITE_P(
    Policies, ChunkElementPolicy3DTest,
    ::testing::Values(
        ChunkElementPolicy3DTestParams{ 10, 5, 5, 5, "Small3DTest" },
        ChunkElementPolicy3DTestParams{ 100, 5, 5, 5, "Medium3DTest" },
        ChunkElementPolicy3DTestParams{ 1000, 5, 5, 5, "Large3DTest" },
        ChunkElementPolicy3DTestParams{ 1024, 5, 5, 5, "ExactChunkSize3DTest" },
        ChunkElementPolicy3DTestParams{ 50, 3, 3, 3, "Small3DTest3x3x3" },
        ChunkElementPolicy3DTestParams{ 50, 4, 4, 4, "Small3DTest4x4x4" }));

// ------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
