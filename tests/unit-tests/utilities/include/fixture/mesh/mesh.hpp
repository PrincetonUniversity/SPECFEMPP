#pragma once

#include "../../../../SPECFEM_Environment.hpp"
#include "../mesh.hpp"
#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "mesh/mesh.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>

namespace specfem::test_fixture {

template <typename Initializer> struct Mesh2D {
  static_assert(
      std::is_base_of_v<MeshInitializer2D::MeshInitializer2D, Initializer>,
      "Mesh2D template parameter expects MeshInitializer2D!");
  static constexpr specfem::dimension::type DimensionTag =
      specfem::dimension::type::dim2;
  static constexpr int ndim = specfem::dimension::dimension<DimensionTag>::dim;

private:
  inline static std::unique_ptr<specfem::mesh::mesh<DimensionTag> >
      _mesh_instance = nullptr;
  inline static int refcount = 0;

  template <typename dummy, typename = void>
  struct is_from_file_t : std::false_type {};
  template <typename dummy>
  struct is_from_file_t<dummy,
                        std::enable_if_t<Initializer::is_from_file, void> >
      : std::true_type {};

public:
  static constexpr bool is_from_file = is_from_file_t<int>::value;

  Mesh2D() {
    if (_mesh_instance == nullptr) {
      if constexpr (is_from_file) {
        specfem::MPI::MPI *mpi = SPECFEMEnvironment::get_mpi();

        _mesh_instance = std::make_unique<specfem::mesh::mesh<DimensionTag> >(
            specfem::io::read_2d_mesh(
                Initializer::get_filename(), specfem::enums::elastic_wave::psv,
                specfem::enums::electromagnetic_wave::te, mpi));

      } else {
        throw std::runtime_error("Mesh2D from internal variables (not from a "
                                 "database file) not yet implemented.");
      }
    }
    ++refcount;
  }
  ~Mesh2D() {
    --refcount;
    if (refcount == 0) {
      _mesh_instance = nullptr;
    }
  }

  const specfem::mesh::mesh<DimensionTag> &mesh_instance() const {
    return *_mesh_instance;
  }
};

namespace MeshInitializer2D {
struct ThreeElementNonconforming : MeshInitializer2D {
  static std::string description() { return "3 element nonconforming grid"; }
  static constexpr specfem::enums::elastic_wave elastic_wave_type =
      specfem::enums::elastic_wave::psv;
  static constexpr specfem::enums::electromagnetic_wave
      electromagnetic_wave_type = specfem::enums::electromagnetic_wave::te;

  static constexpr bool is_from_file = true;
  static std::string get_filename() {
    return "data/dim2/3_elem_nonconforming/database.bin";
  }
};
} // namespace MeshInitializer2D
} // namespace specfem::test_fixture
