#include "../Kokkos_Environment.hpp"
#include "../MPI_environment.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "execution/range_iterator.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "parallel_configuration/range_config.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ------------------------------------------------------------------------
// Test configuration
namespace test_configuration {
struct configuration {
public:
  int processors;
};

void operator>>(YAML::Node &Node, configuration &configuration) {
  configuration.processors = Node["nproc"].as<int>();
  return;
}

struct parameters {
  int nglob;
  int nspec;
};

void operator>>(YAML::Node &Node, parameters &parameters) {
  parameters.nglob = Node["nglob"].as<int>();
  parameters.nspec = Node["nspec"].as<int>();
  return;
}

struct Test {
public:
  Test(const YAML::Node &Node) {
    name = Node["name"].as<std::string>();
    description = Node["description"].as<std::string>();
    YAML::Node config = Node["config"];
    config >> configuration;
    YAML::Node parameters = Node["parameters"];
    parameters >> this->parameters;
    return;
  }

  std::string name;
  std::string description;
  test_configuration::parameters parameters;
  test_configuration::configuration configuration;
};
} // namespace test_configuration

// ------------------------------------------------------------------------
// Reading test config

void parse_test_config(const YAML::Node &yaml,
                       std::vector<test_configuration::Test> &tests) {
  YAML::Node all_tests = yaml["Tests"];
  assert(all_tests.IsSequence());

  for (auto N : all_tests)
    tests.push_back(test_configuration::Test(N));

  return;
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
      "execute_range_policy", iterator, KOKKOS_LAMBDA(const PointIndex &index) {
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

  const specfem::mesh_entity::element<dimension> element_grid(ngllz, ngllx);

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
      KOKKOS_LAMBDA(const PointIndex &point_index) {
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

class POLICIES : public ::testing::Test {
protected:
  class Iterator {
  public:
    Iterator(test_configuration::Test *p_Test, int *p_nglob, int *p_nspec)
        : p_Test(p_Test), p_nglob(p_nglob), p_nspec(p_nspec) {}

    std::tuple<test_configuration::Test, int, int> operator*() {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[RUNNING]\033[0m " << p_Test->name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      return std::make_tuple(*p_Test, *p_nglob, *p_nspec);
    }

    Iterator &operator++() {
      ++p_Test;
      ++p_nglob;
      ++p_nspec;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return p_Test != other.p_Test;
    }

  private:
    test_configuration::Test *p_Test;
    int *p_nglob;
    int *p_nspec;
  };

  POLICIES() {

    std::string config_filename = "policies/test_config.yaml";
    parse_test_config(YAML::LoadFile(config_filename), Tests);

    for (auto &Test : Tests) {
      nglobs.push_back(Test.parameters.nglob);
      nspecs.push_back(Test.parameters.nspec);
    }

    return;
  }

  Iterator begin() { return Iterator(&Tests[0], &nglobs[0], &nspecs[0]); }
  Iterator end() {
    return Iterator(&Tests[Tests.size()], &nglobs[nglobs.size()],
                    &nspecs[nspecs.size()]);
  }

  std::vector<test_configuration::Test> Tests;
  std::vector<int> nglobs;
  std::vector<int> nspecs;
};

TEST_F(POLICIES, RangePolicy) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    const auto nglob = std::get<1>(parameters);
    const auto nspec = std::get<2>(parameters);

    using ParallelConfig = specfem::parallel_config::default_range_config<
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;
    using SimdParallelConfig = specfem::parallel_config::default_range_config<
        specfem::datatype::simd<type_real, true>,
        Kokkos::DefaultExecutionSpace>;

    const auto check_test_view = [&](const auto &test_view, std::string error) {
      for (int iglob = 0; iglob < nglob; iglob++) {
        if (test_view(iglob) != 1) {
          ADD_FAILURE();

          std::cout << "--------------------------------------------------\n"
                    << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
                    << "\n"
                    << "- Error: " << error << "\n"
                    << "  Index: \n "
                    << "    iglob = " << iglob << "\n"
                    << "--------------------------------------------------\n\n"
                    << std::endl;
          return;
        }
      }

      return;
    };

    auto test_view = execute_range_policy<ParallelConfig>(nglob);
    auto simd_test_view = execute_range_policy<SimdParallelConfig>(nglob);

    check_test_view(test_view, "Error in RangePolicy with SIMD OFF");
    check_test_view(simd_test_view, "Error in RangePolicy with SIMD ON");

    std::cout << "--------------------------------------------------\n"
              << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
              << "--------------------------------------------------\n\n"
              << std::endl;
  }
}

TEST_F(POLICIES, ChunkElementPolicy) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    const auto nglob = std::get<1>(parameters);
    const auto nspec = std::get<2>(parameters);

    const int ngllz = 5;
    const int ngllx = 5;

    using ParallelConfig = specfem::parallel_config::default_chunk_config<
        specfem::dimension::type::dim2,
        specfem::datatype::simd<type_real, false>,
        Kokkos::DefaultExecutionSpace>;

    using SimdParallelConfig = specfem::parallel_config::default_chunk_config<
        specfem::dimension::type::dim2,
        specfem::datatype::simd<type_real, true>,
        Kokkos::DefaultExecutionSpace>;

    const auto check_test_view = [&](const auto &test_view, std::string error) {
      for (int ispec = 0; ispec < nspec; ispec++) {
        for (int iz = 0; iz < ngllz; iz++) {
          for (int ix = 0; ix < ngllx; ix++) {
            if (test_view(ispec, iz, ix) != 1) {
              ADD_FAILURE();

              std::cout
                  << "--------------------------------------------------\n"
                  << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
                  << "\n"
                  << "- Error: " << error << "\n"
                  << "  Index: \n "
                  << "    ispec = " << ispec << "\n"
                  << "    iz = " << iz << "\n"
                  << "    ix = " << ix << "\n"
                  << "--------------------------------------------------\n\n"
                  << std::endl;
              return;
            }
          }
        }
      }

      return;
    };

    auto test_view =
        execute_chunk_element_policy<ParallelConfig>(nspec, ngllz, ngllx);
    auto simd_test_view =
        execute_chunk_element_policy<SimdParallelConfig>(nspec, ngllz, ngllx);

    check_test_view(test_view, "Error in ChunkElementPolicy with SIMD OFF");
    check_test_view(simd_test_view, "Error in ChunkElementPolicy with SIMD ON");

    std::cout << "--------------------------------------------------\n"
              << "\033[0;32m[PASSED]\033[0m " << Test.name << "\n"
              << "--------------------------------------------------\n\n"
              << std::endl;
  }
}

// TEST(POLICIES, RangePolicy) {
//   specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();
//   std::string config_filename =
//       "policies/test_config.yaml";
//   std::vector<test_configuration::Test> Tests;
//   parse_test_config(YAML::LoadFile(config_filename), Tests);

//   specfem_data specfem_test(Tests);

//   for (auto [Test, assembly] : specfem_test) {

//     const int nglob = compute_nglob(assembly.mesh.points.h_index_mapping);
//     const int nspec = assembly.mesh.points.nspec;
//     const int ngllz = assembly.mesh.points.ngllz;
//     const int ngllx = assembly.mesh.points.ngllx;
//     const auto index_mapping = assembly.mesh.points.h_index_mapping;

//     using ParallelConfig = specfem::parallel_config::default_range_config<
//         specfem::datatype::simd<type_real, false>,
//         Kokkos::DefaultExecutionSpace>;
//     using SimdParallelConfig =
//     specfem::parallel_config::default_range_config<
//         specfem::datatype::simd<type_real, true>,
//         Kokkos::DefaultExecutionSpace>;

//     const auto check_test_view = [&](const auto &test_view, std::string
//     error) {
//       for (int iglob = 0; iglob < nglob; iglob++) {
//         if (test_view(iglob) != 1) {
//           ADD_FAILURE();

//           std::cout << "--------------------------------------------------\n"
//                     << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
//                     << "\n"
//                     << "- Error: " << error << "\n"
//                     << "  Index: \n "
//                     << "    iglob = " << iglob << "\n"
//                     <<
//                     "--------------------------------------------------\n\n"
//                     << std::endl;
//           return;
//         }
//       }

//       for (int ispec = 0; ispec < nspec; ispec++) {
//         for (int iz = 0; iz < ngllz; iz++) {
//           for (int ix = 0; ix < ngllx; ix++) {
//             const int iglob = index_mapping(ispec, iz, ix);
//             if (test_view(iglob) != 1) {
//               ADD_FAILURE();

//               std::cout
//                   << "--------------------------------------------------\n"
//                   << "\033[0;31m[FAILED]\033[0m Test name: " << Test.name
//                   << "\n"
//                   << "- Error: " << error << "\n"
//                   << "  Index: \n "
//                   << "    ispec = " << ispec << "\n"
//                   << "    iz = " << iz << "\n"
//                   << "    ix = " << ix << "\n"
//                   << "--------------------------------------------------\n\n"
//                   << std::endl;
//               return;
//             }
//           }
//         }
//       }
//     };

//     auto test_view = execute_range_policy<ParallelConfig>(nglob);
//     auto simd_test_view = execute_range_policy<SimdParallelConfig>(nglob);

//     check_test_view(test_view, "Error in RangePolicy with SIMD OFF");
//     check_test_view(simd_test_view, "Error in RangePolicy with SIMD ON");

//     std::cout << "--------------------------------------------------\n"
//               << "\033[0;32m[PASSED]\033[0m Test name: " << Test.name << "\n"
//               << "--------------------------------------------------\n\n"
//               << std::endl;
//   }

//   return;
// }

// ------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
