#include "specfem/assembly/element_intersections.hpp"
#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

// Functors to replace lambdas for CUDA compatibility
// (CUDA does not allow extended __host__ __device__ lambdas inside
// functions with private/protected access, such as TEST_F's TestBody)

template <typename FaceViewType> struct InitializeFacesFunctor {
  FaceViewType view;
  specfem::mesh_entity::dim3::type face_type;
  int num_points;
  int element_multiplier;
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3> elem;

  InitializeFacesFunctor(FaceViewType view_,
                         specfem::mesh_entity::dim3::type face_type_,
                         int num_points_, int element_multiplier_)
      : view(view_), face_type(face_type_), num_points(num_points_),
        element_multiplier(element_multiplier_), elem(num_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    view.element_index(i) = i * element_multiplier;
    view.face_index(i) = i;
    view.face_types(i) = face_type;
    for (int j = 0; j < num_points; ++j) {
      for (int k = 0; k < num_points; ++k) {
        int iz_val, iy_val, ix_val;
        elem.get_face_coordinates(face_type, j, k, iz_val, iy_val, ix_val);
        view.iz(i, j, k) = iz_val;
        view.iy(i, j, k) = iy_val;
        view.ix(i, j, k) = ix_val;
      }
    }
  }
};

template <typename FaceViewType, typename ResultsType>
struct TestSingleFaceFunctor {
  FaceViewType view;
  ResultsType results;

  TestSingleFaceFunctor(FaceViewType view_, ResultsType results_)
      : view(view_), results(results_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int) const {
    auto face = view(1);
    results(0) = face.n_points;
    results(1) = face.element_index;
    results(2) = face.face_index;
  }
};

template <typename FaceViewType, typename ResultsType>
struct TestRangeAccessFaceFunctor {
  FaceViewType view;
  ResultsType results;

  TestRangeAccessFaceFunctor(FaceViewType view_, ResultsType results_)
      : view(view_), results(results_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int) const {
    FaceViewType subview = view(Kokkos::make_pair(1, 3));
    results(0) = subview.N;
    results(1) = subview.n_points;
    results(2) = subview.element_index(0);
    results(3) = subview.element_index(1);
  }
};

template <typename StorageType> struct InitCoords2DFunctor {
  StorageType iz_storage;
  StorageType iy_storage;
  StorageType ix_storage;
  specfem::mesh_entity::dim3::type face_type;
  int num_points;
  specfem::mesh_entity::element<specfem::element::dimension_tag::dim3> elem;

  InitCoords2DFunctor(StorageType iz_, StorageType iy_, StorageType ix_,
                      specfem::mesh_entity::dim3::type face_type_,
                      int num_points_)
      : iz_storage(iz_), iy_storage(iy_), ix_storage(ix_),
        face_type(face_type_), num_points(num_points_), elem(num_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    int iz_val, iy_val, ix_val;
    elem.get_face_coordinates(face_type, i, j, iz_val, iy_val, ix_val);
    iz_storage(i, j) = iz_val;
    iy_storage(i, j) = iy_val;
    ix_storage(i, j) = ix_val;
  }
};

template <typename StorageType, typename ResultsType>
struct TestFaceOperatorFunctor {
  StorageType iz_storage;
  StorageType iy_storage;
  StorageType ix_storage;
  ResultsType results;
  specfem::mesh_entity::dim3::type face_type;
  int num_points;

  TestFaceOperatorFunctor(StorageType iz_, StorageType iy_, StorageType ix_,
                          ResultsType results_,
                          specfem::mesh_entity::dim3::type face_type_,
                          int num_points_)
      : iz_storage(iz_), iy_storage(iy_), ix_storage(ix_), results(results_),
        face_type(face_type_), num_points(num_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int) const {
    using Face = specfem::assembly::Face<Kokkos::DefaultExecutionSpace>;
    using FaceIndex =
        specfem::point::face_index<specfem::element::dimension_tag::dim3>;

    auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL(), Kokkos::ALL());
    auto iy_strided = Kokkos::subview(iy_storage, Kokkos::ALL(), Kokkos::ALL());
    auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL(), Kokkos::ALL());

    Face face(num_points, 42, 3, face_type, iz_strided, iy_strided, ix_strided);
    FaceIndex idx = face(2, 3);

    results(0) = idx.ispec;
    results(1) = idx.iface;
    results(2) = idx.ipoint_i;
    results(3) = idx.ipoint_j;
    results(4) = idx.iz;
    results(5) = idx.iy;
    results(6) = idx.ix;
    results(7) = static_cast<int>(idx.face_type);
  }
};

template <typename StorageType, typename ResultsType>
struct TestFaceTypeFunctor {
  StorageType iz_storage;
  StorageType iy_storage;
  StorageType ix_storage;
  ResultsType results;
  specfem::mesh_entity::dim3::type face_type;
  int num_points;

  TestFaceTypeFunctor(StorageType iz_, StorageType iy_, StorageType ix_,
                      ResultsType results_,
                      specfem::mesh_entity::dim3::type face_type_,
                      int num_points_)
      : iz_storage(iz_), iy_storage(iy_), ix_storage(ix_), results(results_),
        face_type(face_type_), num_points(num_points_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int) const {
    using Face = specfem::assembly::Face<Kokkos::DefaultExecutionSpace>;
    using FaceIndex =
        specfem::point::face_index<specfem::element::dimension_tag::dim3>;

    auto iz_strided = Kokkos::subview(iz_storage, Kokkos::ALL(), Kokkos::ALL());
    auto iy_strided = Kokkos::subview(iy_storage, Kokkos::ALL(), Kokkos::ALL());
    auto ix_strided = Kokkos::subview(ix_storage, Kokkos::ALL(), Kokkos::ALL());

    Face face(num_points, 0, 0, face_type, iz_strided, iy_strided, ix_strided);
    FaceIndex idx = face(2, 3);

    results(0) = idx.iz;
    results(1) = idx.iy;
    results(2) = idx.ix;
  }
};

} // namespace

class AssemblyFaceViewTest : public ::testing::Test {
protected:
  using FaceView = specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;
  using Face = specfem::assembly::Face<Kokkos::DefaultExecutionSpace>;

  static constexpr int num_faces = 6;
  static constexpr int num_points = 5;
};

TEST_F(AssemblyFaceViewTest, DefaultConstructor) {
  FaceView view;
  EXPECT_EQ(view.N, 0);
  EXPECT_EQ(view.n_points, 0);
}

TEST_F(AssemblyFaceViewTest, AllocatingConstructor) {
  FaceView view("test_faces", num_faces, num_points);

  EXPECT_EQ(view.N, num_faces);
  EXPECT_EQ(view.n_points, num_points);

  // Check view dimensions
  EXPECT_EQ(view.element_index.extent(0), num_faces);
  EXPECT_EQ(view.face_index.extent(0), num_faces);
  EXPECT_EQ(view.face_types.extent(0), num_faces);
  EXPECT_EQ(view.iz.extent(0), num_faces);
  EXPECT_EQ(view.iz.extent(1), num_points);
  EXPECT_EQ(view.iz.extent(2), num_points);
  EXPECT_EQ(view.iy.extent(0), num_faces);
  EXPECT_EQ(view.iy.extent(1), num_points);
  EXPECT_EQ(view.iy.extent(2), num_points);
  EXPECT_EQ(view.ix.extent(0), num_faces);
  EXPECT_EQ(view.ix.extent(1), num_points);
  EXPECT_EQ(view.ix.extent(2), num_points);
}

TEST_F(AssemblyFaceViewTest, SingleFaceAccess) {
  FaceView view("test_faces", num_faces, num_points);

  // Initialize face data on device
  constexpr auto bottom = specfem::mesh_entity::dim3::type::bottom;

  Kokkos::parallel_for(
      "initialize_faces",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_faces),
      InitializeFacesFunctor<FaceView>(view, bottom, num_points, 10));
  Kokkos::fence();

  // Test single face access
  Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_single_face", 1,
      TestSingleFaceFunctor<FaceView, decltype(results)>(view, results));
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), num_points);
  EXPECT_EQ(host_results(1), 10); // element_index for face 1
  EXPECT_EQ(host_results(2), 1);  // face_index for face 1
}

TEST_F(AssemblyFaceViewTest, RangeAccess) {
  FaceView view("test_faces", num_faces, num_points);

  // Initialize face data
  constexpr auto top = specfem::mesh_entity::dim3::type::top;

  Kokkos::parallel_for(
      "initialize_faces",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_faces),
      InitializeFacesFunctor<FaceView>(view, top, num_points, 100));
  Kokkos::fence();

  // Test range access [1, 3) should give 2 faces
  Kokkos::View<int[4], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_range_access", 1,
      TestRangeAccessFaceFunctor<FaceView, decltype(results)>(view, results));
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), 2); // 2 faces in range [1, 3)
  EXPECT_EQ(host_results(1), num_points);
  EXPECT_EQ(host_results(2), 100); // element_index for original face 1
  EXPECT_EQ(host_results(3), 200); // element_index for original face 2
}

class AssemblyFaceTest : public ::testing::Test {
protected:
  using Face = specfem::assembly::Face<Kokkos::DefaultExecutionSpace>;
  using IndexView =
      Kokkos::View<int **, Kokkos::LayoutStride, Kokkos::DefaultExecutionSpace>;

  static constexpr int num_points = 5;
};

TEST_F(AssemblyFaceTest, OperatorPointAccess) {
  // Create index views
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> iz_storage(
      "iz", num_points, num_points);
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> iy_storage(
      "iy", num_points, num_points);
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> ix_storage(
      "ix", num_points, num_points);

  constexpr auto left = specfem::mesh_entity::dim3::type::left;
  using StorageType = decltype(iz_storage);

  // Initialize quadrature point coordinates
  Kokkos::parallel_for(
      "init_coords",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
          { 0, 0 }, { num_points, num_points }),
      InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage, left,
                                       num_points));
  Kokkos::fence();

  // Test Face operator()
  using FaceIndex =
      specfem::point::face_index<specfem::element::dimension_tag::dim3>;
  Kokkos::View<int[8], Kokkos::DefaultExecutionSpace> results("results");

  Kokkos::parallel_for(
      "test_face_operator", 1,
      TestFaceOperatorFunctor<StorageType, decltype(results)>(
          iz_storage, iy_storage, ix_storage, results, left, num_points));
  Kokkos::fence();

  auto host_results = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(host_results, results);

  EXPECT_EQ(host_results(0), 42); // element_index
  EXPECT_EQ(host_results(1), 3);  // face_index
  EXPECT_EQ(host_results(2), 2);  // ipoint_i
  EXPECT_EQ(host_results(3), 3);  // ipoint_j
  EXPECT_EQ(host_results(4), 2);  // iz for left face (iz = ipoint_i)
  EXPECT_EQ(host_results(5), 3);  // iy for left face (iy = ipoint_j)
  EXPECT_EQ(host_results(6), 0);  // ix for left face (always 0)
  EXPECT_EQ(host_results(7), static_cast<int>(left));
}

TEST_F(AssemblyFaceTest, AllFaceTypes) {
  // Test all six 3D face types
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> iz_storage(
      "iz", num_points, num_points);
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> iy_storage(
      "iy", num_points, num_points);
  Kokkos::View<int **, Kokkos::DefaultExecutionSpace> ix_storage(
      "ix", num_points, num_points);

  using FaceIndex =
      specfem::point::face_index<specfem::element::dimension_tag::dim3>;
  using StorageType = decltype(iz_storage);

  // Test bottom face (z = 0)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::bottom;
    Kokkos::parallel_for(
        "init_bottom",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_bottom", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 0); // iz = 0 for bottom
    EXPECT_EQ(host_results(1), 2); // iy = ipoint_i
    EXPECT_EQ(host_results(2), 3); // ix = ipoint_j
  }

  // Test top face (z = ngll-1)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::top;
    Kokkos::parallel_for(
        "init_top",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_top", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), num_points - 1); // iz = ngll-1 for top
    EXPECT_EQ(host_results(1), 2);              // iy = ipoint_i
    EXPECT_EQ(host_results(2), 3);              // ix = ipoint_j
  }

  // Test left face (x = 0)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::left;
    Kokkos::parallel_for(
        "init_left",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_left", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 2); // iz = ipoint_i for left
    EXPECT_EQ(host_results(1), 3); // iy = ipoint_j for left
    EXPECT_EQ(host_results(2), 0); // ix = 0 for left
  }

  // Test right face (x = ngll-1)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::right;
    Kokkos::parallel_for(
        "init_right",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_right", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 2);              // iz = ipoint_i for right
    EXPECT_EQ(host_results(1), 3);              // iy = ipoint_j for right
    EXPECT_EQ(host_results(2), num_points - 1); // ix = ngll-1 for right
  }

  // Test front face (y = 0)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::front;
    Kokkos::parallel_for(
        "init_front",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_front", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 2); // iz = ipoint_i for front
    EXPECT_EQ(host_results(1), 0); // iy = 0 for front
    EXPECT_EQ(host_results(2), 3); // ix = ipoint_j for front
  }

  // Test back face (y = ngll-1)
  {
    constexpr auto face_type = specfem::mesh_entity::dim3::type::back;
    Kokkos::parallel_for(
        "init_back",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
            { 0, 0 }, { num_points, num_points }),
        InitCoords2DFunctor<StorageType>(iz_storage, iy_storage, ix_storage,
                                         face_type, num_points));
    Kokkos::fence();

    Kokkos::View<int[3], Kokkos::DefaultExecutionSpace> results("results");
    Kokkos::parallel_for("test_back", 1,
                         TestFaceTypeFunctor<StorageType, decltype(results)>(
                             iz_storage, iy_storage, ix_storage, results,
                             face_type, num_points));
    Kokkos::fence();

    auto host_results = Kokkos::create_mirror_view(results);
    Kokkos::deep_copy(host_results, results);
    EXPECT_EQ(host_results(0), 2);              // iz = ipoint_i for back
    EXPECT_EQ(host_results(1), num_points - 1); // iy = ngll-1 for back
    EXPECT_EQ(host_results(2), 3);              // ix = ipoint_j for back
  }
}

class AssemblyFaceViewHostMirrorTest : public ::testing::Test {
protected:
  using FaceView = specfem::assembly::FaceView<Kokkos::DefaultExecutionSpace>;

  static constexpr int num_faces = 3;
  static constexpr int num_points = 4;
};

TEST_F(AssemblyFaceViewHostMirrorTest, HostMirrorType) {
  // Verify HostMirror type is correctly defined
  using HostMirror = typename FaceView::HostMirror;

  HostMirror host_view("host_faces", num_faces, num_points);

  EXPECT_EQ(host_view.N, num_faces);
  EXPECT_EQ(host_view.n_points, num_points);

  // Initialize on host
  for (int i = 0; i < num_faces; ++i) {
    host_view.element_index(i) = i;
    host_view.face_index(i) = i * 2;
    host_view.face_types(i) = specfem::mesh_entity::dim3::type::bottom;
    for (int j = 0; j < num_points; ++j) {
      for (int k = 0; k < num_points; ++k) {
        host_view.iz(i, j, k) = 0;
        host_view.iy(i, j, k) = j;
        host_view.ix(i, j, k) = k;
      }
    }
  }

  // Verify values
  EXPECT_EQ(host_view.element_index(1), 1);
  EXPECT_EQ(host_view.face_index(2), 4);
  EXPECT_EQ(host_view.iy(0, 2, 3), 2);
  EXPECT_EQ(host_view.ix(0, 2, 3), 3);
}
