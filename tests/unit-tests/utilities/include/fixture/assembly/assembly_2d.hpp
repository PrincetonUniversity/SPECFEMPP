#pragma once

#include "../assembly.hpp"
#include "../mesh.hpp"
#include "specfem/assembly.hpp"

#include <type_traits>

namespace specfem::test_fixture {
template <typename Initializer> struct Assembly2D {
  static_assert(std::is_base_of_v<AssemblyInitializer2D::AssemblyInitializer2D,
                                  Initializer>,
                "Assembly2D template parameter expects AssemblyInitializer2D!");
  static constexpr specfem::dimension::type DimensionTag =
      specfem::dimension::type::dim2;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

private:
  inline static std::unique_ptr<specfem::assembly::assembly<DimensionTag> >
      _assembly_instance = nullptr;
  inline static int refcount = 0;

public:
  Assembly2D() {
    if (_assembly_instance == nullptr) {
      _assembly_instance =
          std::make_unique<specfem::assembly::assembly<DimensionTag> >(
              Initializer::generate_assembly());
    }
    ++refcount;
  }
  ~Assembly2D() {
    --refcount;
    if (refcount == 0) {
      _assembly_instance = nullptr;
    }
  }

  const specfem::assembly::assembly<DimensionTag> &assembly_instance() const {
    return *_assembly_instance;
  }
};

namespace AssemblyInitializer2D {
static constexpr specfem::dimension::type DimensionTag =
    specfem::dimension::type::dim2;
static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

template <typename MeshInitializer> struct FromMesh : AssemblyInitializer2D {
  static_assert(
      std::is_base_of_v<MeshInitializer2D::MeshInitializer2D, MeshInitializer>,
      "FromMesh template parameter expects MeshInitializer2D!");

  static specfem::assembly::assembly<DimensionTag> generate_assembly() {
    Mesh2D<MeshInitializer> mesh;

    // TODO: consider how to handle other parameters
    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    std::vector<std::shared_ptr<
        specfem::sources::source<specfem::dimension::type::dim2> > >
        sources;
    std::vector<std::shared_ptr<
        specfem::receivers::receiver<specfem::dimension::type::dim2> > >
        receivers;
    return specfem::assembly::assembly<specfem::dimension::type::dim2>(
        mesh.mesh_instance(), quadrature, sources, receivers, {}, 1.0, 0.0, 1,
        1, 1, specfem::simulation::type::forward, false, nullptr);
  }
};
} // namespace AssemblyInitializer2D
} // namespace specfem::test_fixture
