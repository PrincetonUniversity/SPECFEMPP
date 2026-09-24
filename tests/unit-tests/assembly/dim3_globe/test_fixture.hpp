#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/attenuation.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/mesh.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace specfem::test_configuration {

struct GlobeAssembly3D {
  constexpr static auto dimension = specfem::element::dimension_tag::dim3;

  specfem::assembly::assembly<dimension> assembly;

  explicit GlobeAssembly3D(const std::string &database_file) {
    const auto mesh = specfem::io::read_globe_mesh(
        database_file, specfem::attenuation::Setup{});
    const specfem::quadrature::quadratures quadratures(
        specfem::quadrature::gll::gll{});
    std::vector<std::shared_ptr<specfem::sources::source<dimension>>> sources;
    std::vector<std::shared_ptr<specfem::receivers::receiver<dimension>>>
        receivers;
    const std::vector<specfem::enums::wavefield> seismogram_types;
    const std::shared_ptr<specfem::io::reader> property_reader;

    assembly = specfem::assembly::assembly<dimension>(
        mesh, quadratures, sources, receivers, seismogram_types, 0.0, 0.01, 1,
        1, 1, specfem::simulation::type::forward, false, property_reader);
  }
};

} // namespace specfem::test_configuration

class GlobeAssembly3DTest : public ::testing::Test {
protected:
  void SetUp() override {
    assembly = std::make_unique<specfem::test_configuration::GlobeAssembly3D>(
        "data/dim3_globe/GlobalSmallMesh/DATABASES_MPI/"
        "proc000000_specfempp_database.bin");
  }

  std::unique_ptr<specfem::test_configuration::GlobeAssembly3D> assembly;
};
