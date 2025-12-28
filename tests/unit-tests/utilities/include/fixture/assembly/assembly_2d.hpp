#pragma once

#include "../assembly.hpp"
#include "../impl/descriptions.hpp"
#include "../mesh.hpp"
#include "specfem/assembly.hpp"

namespace specfem::test_fixture {
template <typename Initializer> struct Assembly2D {
  static_assert(std::is_base_of_v<AssemblyInitializer2D::AssemblyInitializer2D,
                                  Initializer>,
                "Assembly2D template parameter expects AssemblyInitializer2D!");
  using AssemblyInitializer = Initializer;
  static constexpr specfem::dimension::type DimensionTag =
      specfem::dimension::type::dim2;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

private:
  inline static std::shared_ptr<specfem::assembly::assembly<DimensionTag> >
      _global_assembly_instance = nullptr;

  std::shared_ptr<specfem::assembly::assembly<DimensionTag> >
      _assembly_instance;

public:
  Assembly2D() {
    if (_global_assembly_instance == nullptr) {
      _global_assembly_instance =
          std::make_shared<specfem::assembly::assembly<DimensionTag> >(
              Initializer::generate_assembly());
    }
    _assembly_instance = _global_assembly_instance;
  }
  ~Assembly2D() {
    _assembly_instance = nullptr;
    if (_global_assembly_instance.use_count() == 1) {
      _global_assembly_instance = nullptr;
    }
  }

  std::shared_ptr<const specfem::assembly::assembly<DimensionTag> >
  assembly_instance() const {
    return _global_assembly_instance;
  }

  static std::string description(const int &indent = 0) {
    return specfem::test_fixture::impl::description<Initializer>::get(indent);
  }
  static std::string name() {
    return specfem::test_fixture::impl::name<Initializer>::get();
  }
};

namespace AssemblyInitializer2D {
static constexpr specfem::dimension::type DimensionTag =
    specfem::dimension::type::dim2;
static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

template <typename Initializer> struct FromMesh : AssemblyInitializer2D {
  static_assert(
      std::is_base_of_v<MeshInitializer2D::MeshInitializer2D, Initializer>,
      "FromMesh template parameter expects MeshInitializer2D!");
  using MeshInitializer = Initializer;

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
        *mesh.mesh_instance(), quadrature, sources, receivers, {}, 1.0, 0.0, 1,
        1, 1, specfem::simulation::type::forward, false, nullptr);
  }
  static std::string description() {
    return std::string("Mesh-initialized assembly:\n  name: ") +
           specfem::test_fixture::impl::name<Initializer>::get() +
           "\n  description:\n" +
           specfem::test_fixture::impl::description<Initializer>::get(4);
  }
  static std::string name() {
    return std::string("FromMesh(") +
           specfem::test_fixture::impl::name<Initializer>::get() + ")";
  }
};
} // namespace AssemblyInitializer2D
} // namespace specfem::test_fixture
