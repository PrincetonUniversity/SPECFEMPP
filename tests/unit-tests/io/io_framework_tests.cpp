#include "../SPECFEM_Environment.hpp"
#include <Kokkos_Core.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>

// Include all I/O framework headers
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/NPY/NPY.hpp"
#include "io/NPZ/NPZ.hpp"
#include "io/operators.hpp"

// Test utilities
#include <type_traits>
#include <vector>

namespace fs = boost::filesystem;

// Helper to get write type from read type
template <typename ReadType> struct GetWriteType;

template <> struct GetWriteType<specfem::io::ASCII<specfem::io::read> > {
  using type = specfem::io::ASCII<specfem::io::write>;
};
template <> struct GetWriteType<specfem::io::HDF5<specfem::io::read> > {
  using type = specfem::io::HDF5<specfem::io::write>;
};
template <> struct GetWriteType<specfem::io::ADIOS2<specfem::io::read> > {
  using type = specfem::io::ADIOS2<specfem::io::write>;
};
template <> struct GetWriteType<specfem::io::NPY<specfem::io::read> > {
  using type = specfem::io::NPY<specfem::io::write>;
};
template <> struct GetWriteType<specfem::io::NPZ<specfem::io::read> > {
  using type = specfem::io::NPZ<specfem::io::write>;
};

// Base test class
class IOFrameworkTestBase : public ::testing::Test {
protected:
  // Generate unique filename based on I/O type and test name to avoid conflicts
  // during parallel execution
  template <typename IOType>
  std::string getTestFileName(const std::string &test_name) {
    std::string base_name;
    if constexpr (std::is_same_v<IOType,
                                 specfem::io::ASCII<specfem::io::write> >) {
      base_name = "test_ascii_write";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::ASCII<specfem::io::read> >) {
      base_name = "test_ascii_read";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::HDF5<specfem::io::write> >) {
      base_name = "test_hdf5_write";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::HDF5<specfem::io::read> >) {
      base_name = "test_hdf5_read";
    } else if constexpr (std::is_same_v<IOType, specfem::io::ADIOS2<
                                                    specfem::io::write> >) {
      base_name = "test_adios2_write";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::ADIOS2<specfem::io::read> >) {
      base_name = "test_adios2_read";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::NPY<specfem::io::write> >) {
      base_name = "test_npy_write";
    } else if constexpr (std::is_same_v<IOType,
                                        specfem::io::NPY<specfem::io::read> >) {
      base_name = "test_npy_read";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::NPZ<specfem::io::write> >) {
      base_name = "test_npz_write";
    } else if constexpr (std::is_same_v<IOType,
                                        specfem::io::NPZ<specfem::io::read> >) {
      base_name = "test_npz_read";
    } else {
      base_name = "test_unknown";
    }
    return base_name + "_" + test_name;
  }

  // Clean up files/directories for a specific test filename
  template <typename IOType> void cleanup(const std::string &test_name) {
    std::string base_name = getTestFileName<IOType>(test_name);
    std::vector<std::string> patterns = { base_name, base_name + ".h5",
                                          base_name + ".bp" };

    for (const auto &pattern : patterns) {
      fs::path path(pattern);
      if (fs::exists(path)) {
        if (fs::is_directory(path)) {
          fs::remove_all(path);
        } else {
          fs::remove(path);
        }
      }
    }
  }

  // Generate test data
  template <typename T> std::vector<T> generateTestData(size_t size) {
    std::vector<T> data(size);

    if constexpr (std::is_same_v<T, bool>) {
      // Special case for bool - create alternating pattern
      for (size_t i = 0; i < size; ++i) {
        data[i] = (i % 2 == 0);
      }
    } else if constexpr (std::is_integral_v<T>) {
      // Integer types
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i * 7 + 42); // Deterministic pattern
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      // Floating point types
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i * 3.14159 + 2.718);
      }
    }

    return data;
  }

  // Compare two vectors with tolerance for floating point
  template <typename T>
  bool compareData(const std::vector<T> &expected, const std::vector<T> &actual,
                   double tolerance = 1e-10) {
    if (expected.size() != actual.size())
      return false;

    for (size_t i = 0; i < expected.size(); ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
          return false;
        }
      } else {
        if (expected[i] != actual[i]) {
          return false;
        }
      }
    }
    return true;
  }
};

// Test fixture template
template <typename IOType> class IOFrameworkTest : public IOFrameworkTestBase {
protected:
  void SetUp() override {
    std::string test_name = getCurrentTestName();
    cleanup<IOType>(test_name); // Clean up any leftover files
  }

  void TearDown() override {
    std::string test_name = getCurrentTestName();
    cleanup<IOType>(test_name); // Clean up after test
  }

  std::string getTestFile() {
    return this->template getTestFileName<IOType>(getCurrentTestName());
  }

private:
  std::string getCurrentTestName() {
    const ::testing::TestInfo *const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    return test_info->name();
  }
};

// Type list for all I/O frameworks - conditionally include based on available
// packages
using IOTypes = ::testing::Types<
    specfem::io::ASCII<specfem::io::write>,
    specfem::io::ASCII<specfem::io::read>, specfem::io::NPY<specfem::io::write>,
    specfem::io::NPY<specfem::io::read>
#ifndef NO_NPZ
    ,
    specfem::io::NPZ<specfem::io::write>, specfem::io::NPZ<specfem::io::read>
#endif
#ifndef NO_HDF5
    ,
    specfem::io::HDF5<specfem::io::write>, specfem::io::HDF5<specfem::io::read>
#endif
#ifndef NO_ADIOS2
    ,
    specfem::io::ADIOS2<specfem::io::write>,
    specfem::io::ADIOS2<specfem::io::read>
#endif
    >;

TYPED_TEST_SUITE(IOFrameworkTest, IOTypes);

// Test 1: Basic file operations
TYPED_TEST(IOFrameworkTest, BasicFileOperations) {
  using IOType = TypeParam;

  // Test file creation (write operations)
  if constexpr (std::is_same_v<typename IOType::IO_OpType,
                               specfem::io::write>) {
    typename IOType::File file(this->getTestFile());
    EXPECT_NO_THROW(file.flush());

    // Verify file/directory was created
    std::string expected_name = this->getTestFile();
    if constexpr (std::is_same_v<IOType,
                                 specfem::io::HDF5<specfem::io::write> >) {
      expected_name += ".h5";
    } else if constexpr (std::is_same_v<IOType, specfem::io::ADIOS2<
                                                    specfem::io::write> >) {
      expected_name += ".bp";
    } else if constexpr (std::is_same_v<
                             IOType, specfem::io::NPZ<specfem::io::write> >) {
      expected_name += ".npz";
    }

    EXPECT_TRUE(fs::exists(expected_name));
  }
}

// Test 2: Group operations
TYPED_TEST(IOFrameworkTest, GroupOperations) {
  using IOType = TypeParam;

  if constexpr (std::is_same_v<typename IOType::IO_OpType,
                               specfem::io::write>) {
    // Create file and groups
    typename IOType::File file(this->getTestFile());
    auto group1 = file.createGroup("group1");
    auto group2 = file.createGroup("group2");
    auto nested_group = group1.createGroup("nested");
    EXPECT_NO_THROW(file.flush());
  } else {
    // For read operations, first create the file with groups
    {
      using WriteIOType = typename GetWriteType<IOType>::type;
      typename WriteIOType::File write_file(this->getTestFile());
      auto group1 = write_file.createGroup("group1");
      auto group2 = write_file.createGroup("group2");
      auto nested_group = group1.createGroup("nested");
      write_file.flush();
    }

    // Then read it back
    typename IOType::File file(this->getTestFile());
    auto group1 = file.openGroup("group1");
    auto group2 = file.openGroup("group2");
    auto nested_group = group1.openGroup("nested");
    // If we get here without exceptions, the groups were opened successfully
  }
}

// Test 3: Dataset operations with different data types
TYPED_TEST(IOFrameworkTest, DatasetOperations) {
  using IOType = TypeParam;

  const size_t data_size = 100;

  // Test with int data
  auto int_data = this->template generateTestData<int>(data_size);
  Kokkos::View<int *, Kokkos::HostSpace> int_view("test_int", data_size);
  for (size_t i = 0; i < data_size; ++i) {
    int_view(i) = int_data[i];
  }

  // Test with double data
  auto double_data = this->template generateTestData<double>(data_size);
  Kokkos::View<double *, Kokkos::HostSpace> double_view("test_double",
                                                        data_size);
  for (size_t i = 0; i < data_size; ++i) {
    double_view(i) = double_data[i];
  }

  // Test with float data
  auto float_data = this->template generateTestData<float>(data_size);
  Kokkos::View<float *, Kokkos::HostSpace> float_view("test_float", data_size);
  for (size_t i = 0; i < data_size; ++i) {
    float_view(i) = float_data[i];
  }

  if constexpr (std::is_same_v<typename IOType::IO_OpType,
                               specfem::io::write>) {
    // Write operations - create file and datasets
    typename IOType::File file(this->getTestFile());

    // Create datasets
    auto int_dataset =
        file.template createDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
            "int_data", int_view);
    auto double_dataset =
        file.template createDataset<Kokkos::View<double *, Kokkos::HostSpace> >(
            "double_data", double_view);
    auto float_dataset =
        file.template createDataset<Kokkos::View<float *, Kokkos::HostSpace> >(
            "float_data", float_view);

    // Write data
    EXPECT_NO_THROW(int_dataset.write());
    EXPECT_NO_THROW(double_dataset.write());
    EXPECT_NO_THROW(float_dataset.write());

    EXPECT_NO_THROW(file.flush());
  } else {
    // For read operations, first create the file with data
    {
      using WriteIOType = typename GetWriteType<IOType>::type;
      typename WriteIOType::File write_file(this->getTestFile());
      auto int_dataset =
          write_file
              .template createDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
                  "int_data", int_view);
      auto double_dataset = write_file.template createDataset<
          Kokkos::View<double *, Kokkos::HostSpace> >("double_data",
                                                      double_view);
      auto float_dataset = write_file.template createDataset<
          Kokkos::View<float *, Kokkos::HostSpace> >("float_data", float_view);

      int_dataset.write();
      double_dataset.write();
      float_dataset.write();
      write_file.flush();
    }

    // Read data back and verify
    // Read back and verify
    typename IOType::File file(this->getTestFile());

    // Create views for reading
    Kokkos::View<int *, Kokkos::HostSpace> read_int_view("read_int", data_size);
    Kokkos::View<double *, Kokkos::HostSpace> read_double_view("read_double",
                                                               data_size);
    Kokkos::View<float *, Kokkos::HostSpace> read_float_view("read_float",
                                                             data_size);

    // Open datasets and read
    auto int_dataset =
        file.template openDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
            "int_data", read_int_view);
    auto double_dataset =
        file.template openDataset<Kokkos::View<double *, Kokkos::HostSpace> >(
            "double_data", read_double_view);
    auto float_dataset =
        file.template openDataset<Kokkos::View<float *, Kokkos::HostSpace> >(
            "float_data", read_float_view);

    EXPECT_NO_THROW(int_dataset.read());
    EXPECT_NO_THROW(double_dataset.read());
    EXPECT_NO_THROW(float_dataset.read());

    // Verify data
    std::vector<int> read_int_data(data_size);
    std::vector<double> read_double_data(data_size);
    std::vector<float> read_float_data(data_size);

    for (size_t i = 0; i < data_size; ++i) {
      read_int_data[i] = read_int_view(i);
      read_double_data[i] = read_double_view(i);
      read_float_data[i] = read_float_view(i);
    }

    EXPECT_TRUE(this->compareData(int_data, read_int_data));
    EXPECT_TRUE(this->compareData(double_data, read_double_data));
    EXPECT_TRUE(this->compareData(float_data, read_float_data,
                                  1e-6)); // Float precision
  }
}

// Test 4: Bool conversion (especially important for ADIOS2)
TYPED_TEST(IOFrameworkTest, BoolDataOperations) {
  using IOType = TypeParam;

  const size_t data_size = 50;
  auto bool_data = this->template generateTestData<bool>(data_size);
  Kokkos::View<bool *, Kokkos::HostSpace> bool_view("test_bool", data_size);
  for (size_t i = 0; i < data_size; ++i) {
    bool_view(i) = bool_data[i];
  }

  if constexpr (std::is_same_v<typename IOType::IO_OpType,
                               specfem::io::write>) {
    typename IOType::File file(this->getTestFile());
    auto bool_dataset =
        file.template createDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
            "bool_data", bool_view);
    EXPECT_NO_THROW(bool_dataset.write());
    EXPECT_NO_THROW(file.flush());
  } else {
    // Create file with bool data
    {
      using WriteIOType = typename GetWriteType<IOType>::type;
      typename WriteIOType::File write_file(this->getTestFile());
      auto bool_dataset =
          write_file
              .template createDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
                  "bool_data", bool_view);
      bool_dataset.write();
      write_file.flush();
    }

    // Read back and verify
    typename IOType::File file(this->getTestFile());
    Kokkos::View<bool *, Kokkos::HostSpace> read_bool_view("read_bool",
                                                           data_size);
    auto bool_dataset =
        file.template openDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
            "bool_data", read_bool_view);
    EXPECT_NO_THROW(bool_dataset.read());

    // Verify data
    std::vector<bool> read_bool_data(data_size);
    for (size_t i = 0; i < data_size; ++i) {
      read_bool_data[i] = read_bool_view(i);
    }

    EXPECT_TRUE(this->compareData(bool_data, read_bool_data));
  }
}

// Test 5: Complex workflow with groups and datasets
TYPED_TEST(IOFrameworkTest, ComplexWorkflow) {
  using IOType = TypeParam;

  const size_t small_data_size = 25;
  const size_t large_data_size = 200;

  std::cout << "==============================================================="
            << std::endl;
  std::cout << "START" << std::endl;
  std::cout << "==============================================================="
            << std::endl;
  if constexpr (std::is_same_v<typename IOType::IO_OpType,
                               specfem::io::write>) {
    typename IOType::File file(this->getTestFile());

    // Print file name
    std::cout << "WRITE: File name: " << this->getTestFile() << std::endl;

    // Create hierarchical structure
    auto physics_group = file.createGroup("physics");
    auto mesh_group = file.createGroup("mesh");
    auto results_group = physics_group.createGroup("results");

    std::cout << "WRITE: Created groups." << std::endl;

    // Create various datasets in different groups
    auto int_data = this->template generateTestData<int>(small_data_size);
    auto double_data = this->template generateTestData<double>(large_data_size);
    auto bool_data = this->template generateTestData<bool>(small_data_size);

    std::cout << "WRITE: Created test data." << std::endl;

    Kokkos::View<int *, Kokkos::HostSpace> int_view("mesh_ids",
                                                    small_data_size);
    Kokkos::View<double *, Kokkos::HostSpace> double_view("coordinates",
                                                          large_data_size);
    Kokkos::View<bool *, Kokkos::HostSpace> bool_view("active_elements",
                                                      small_data_size);

    std::cout << "WRITE: Created views." << std::endl;

    for (size_t i = 0; i < small_data_size; ++i) {
      int_view(i) = int_data[i];
      double_view(i) = double_data[i];
    }
    std::cout << "WRITE: Populated int/double view." << std::endl;

    for (volatile size_t i = 0; i < small_data_size; ++i) {
      bool_view(i) = bool_data[i];
    }
    std::cout << "WRITE: Populated bool view." << std::endl;

    // Write datasets to different groups
    auto mesh_ids_dataset =
        mesh_group
            .template createDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
                "element_ids", int_view);

    auto coordinates_dataset =
        mesh_group
            .template createDataset<Kokkos::View<double *, Kokkos::HostSpace> >(
                "node_coordinates", double_view);
    auto active_dataset =
        results_group
            .template createDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
                "active_elements", bool_view);

    std::cout << "Created datasets." << std::endl;

    EXPECT_NO_THROW(mesh_ids_dataset.write());
    EXPECT_NO_THROW(coordinates_dataset.write());
    EXPECT_NO_THROW(active_dataset.write());

    std::cout << "WRITE: Wrote datasets." << std::endl;

    EXPECT_NO_THROW(file.flush());

    std::cout << "WRITE: Flushed file." << std::endl;

  } else {
    // Create the complex structure first
    {
      using WriteIOType = typename GetWriteType<IOType>::type;
      typename WriteIOType::File write_file(this->getTestFile());

      std::cout << "READ: Creating complex structure in file: "
                << this->getTestFile() << std::endl;

      auto physics_group = write_file.createGroup("physics");
      auto mesh_group = write_file.createGroup("mesh");
      auto results_group = physics_group.createGroup("results");

      std::cout << "READ: Created groups." << std::endl;

      auto int_data = this->template generateTestData<int>(small_data_size);
      auto double_data =
          this->template generateTestData<double>(large_data_size);
      auto bool_data = this->template generateTestData<bool>(small_data_size);

      std::cout << "READ: Created test data." << std::endl;

      Kokkos::View<int *, Kokkos::HostSpace> int_view("mesh_ids",
                                                      small_data_size);
      Kokkos::View<double *, Kokkos::HostSpace> double_view("coordinates",
                                                            large_data_size);
      Kokkos::View<bool *, Kokkos::HostSpace> bool_view("active_elements",
                                                        small_data_size);

      std::cout << "READ: Created views." << std::endl;

      for (size_t i = 0; i < small_data_size; ++i) {
        int_view(i) = int_data[i];
        double_view(i) = double_data[i];
      }

      std::cout << "READ: Populated int/double view." << std::endl;

      for (volatile size_t i = 0; i < small_data_size; ++i) {
        bool_view(i) = bool_data[i];
      }

      std::cout << "READ: Populated bool view." << std::endl;

      auto mesh_ids_dataset =
          mesh_group
              .template createDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
                  "element_ids", int_view);
      auto coordinates_dataset = mesh_group.template createDataset<
          Kokkos::View<double *, Kokkos::HostSpace> >("node_coordinates",
                                                      double_view);
      auto active_dataset =
          results_group
              .template createDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
                  "active_elements", bool_view);

      std::cout << "READ: Created datasets." << std::endl;

      mesh_ids_dataset.write();
      coordinates_dataset.write();
      active_dataset.write();
      write_file.flush();

      std::cout << "Wrote datasets and flushed file." << std::endl;
    }

    // Read back the complex structure
    auto should_not_throw = [this]() {
      typename IOType::File file(this->getTestFile());

      std::cout << "READ: Opening file: " << this->getTestFile() << std::endl;

      auto physics_group = file.openGroup("physics");
      auto mesh_group = file.openGroup("mesh");
      auto results_group = physics_group.openGroup("results");

      std::cout << "READ: Opened groups." << std::endl;

      Kokkos::View<int *, Kokkos::HostSpace> read_int_view("read_mesh_ids",
                                                           small_data_size);
      Kokkos::View<double *, Kokkos::HostSpace> read_double_view(
          "read_coordinates", large_data_size);
      Kokkos::View<bool *, Kokkos::HostSpace> read_bool_view("read_active",
                                                             small_data_size);

      std::cout << "READ: Created views." << std::endl;

      auto mesh_ids_dataset =
          mesh_group
              .template openDataset<Kokkos::View<int *, Kokkos::HostSpace> >(
                  "element_ids", read_int_view);
      auto coordinates_dataset =
          mesh_group
              .template openDataset<Kokkos::View<double *, Kokkos::HostSpace> >(
                  "node_coordinates", read_double_view);
      auto active_dataset =
          results_group
              .template openDataset<Kokkos::View<bool *, Kokkos::HostSpace> >(
                  "active_elements", read_bool_view);

      std::cout << "READ: Opened datasets." << std::endl;

      mesh_ids_dataset.read();
      coordinates_dataset.read();
      active_dataset.read();

      std::cout << "READ: Read datasets." << std::endl;

      // Basic verification that read succeeded
      EXPECT_GT(read_int_view.extent(0), 0);
      EXPECT_GT(read_double_view.extent(0), 0);
      EXPECT_GT(read_bool_view.extent(0), 0);

      std::cout << "READ: Finished reading datasets." << std::endl;
    };

    // Ensure no exceptions are thrown during read
    EXPECT_NO_THROW(should_not_throw());
  }

  std::cout << "==============================================================="
            << std::endl;
  std::cout << "FINISHED" << std::endl;
  std::cout << "==============================================================="
            << std::endl;
}

// Main test runner
int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  int result = RUN_ALL_TESTS();
  return result;
}
