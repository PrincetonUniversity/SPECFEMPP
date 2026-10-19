#include "../acoustic_elastic.hpp"
#include "specfem/io.hpp"
#include <gtest/gtest.h>

void test_nonconforming_mesh(const std::string &database_file) {

  const auto mesh =
      specfem::io::read_3d_mesh(database_file, specfem::attenuation::Setup{});

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::element::dimension_tag::dim3>>>
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::element::dimension_tag::dim3>>>
      receivers;

  specfem::element_coupling::flux_scheme_configuration flux_scheme_config;
  flux_scheme_config.set_interfacial_meshing_type(
      specfem::element_coupling::interfacial_meshing_type::self_host);

  specfem::assembly::assembly<specfem::element::dimension_tag::dim3> assembly(
      mesh, quadrature, sources, receivers, {}, 1.0, 0.0, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr, flux_scheme_config);

  //   test_nonconforming_container_transfers(assembly);
  specfem::nonconforming_test::kernel::test_nonconforming_acoustic_elastic(
      assembly, database_file);

  specfem::nonconforming_test::kernel::
      test_nonconforming_acoustic_elastic_conjugate(assembly, database_file);
}

TEST(NonconformingKernel3D, acoustic_elastic_11_7) {
  test_nonconforming_mesh(
      "data/dim3/interfaces_only/acoustic_elastic_11-7/database.bin");
}

TEST(NonconformingKernel3D, acoustic_elastic_13_10) {
  test_nonconforming_mesh(
      "data/dim3/interfaces_only/acoustic_elastic_13-10/database.bin");
}
