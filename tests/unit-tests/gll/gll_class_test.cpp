#include "../SPECFEM_Environment.hpp"
TEST(gll_class, CHECK_LENGTHS) {
  gll::gll gll_instance = gll::gll();
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll =
      gll_instance.get_xigll();
  EXPECT_EQ(xigll.extent(0), 5);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
